import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import AgentReflex, LogEvent, SoftSensorModel, SOPDocument
from coreason_signal.soft_sensor.engine import SoftSensorEngine


@pytest.fixture  # type: ignore[misc]
def mock_embedding_model() -> Generator[MagicMock, None, None]:
    with patch("coreason_signal.edge_agent.vector_store.TextEmbedding") as mock:
        instance = mock.return_value
        # Mock embed to return a dummy vector (list of generators)
        instance.embed.side_effect = lambda docs: (np.random.rand(384).tolist() for _ in docs)
        yield instance


@pytest.fixture  # type: ignore[misc]
def mock_onnx_session() -> Generator[MagicMock, None, None]:
    with patch("coreason_signal.soft_sensor.engine.ort.InferenceSession") as mock:
        instance = mock.return_value
        # Default behavior, can be overridden by test
        input_mock = MagicMock()
        input_mock.name = "input"
        output_mock = MagicMock()
        output_mock.name = "output"
        instance.get_inputs.return_value = [input_mock]
        instance.get_outputs.return_value = [output_mock]
        yield instance


def test_reactor_overheat_recovery_loop(
    mock_embedding_model: MagicMock, mock_onnx_session: MagicMock, tmp_path: Path
) -> None:
    """
    COMPLEX SMOKE TEST: 'The Reactor Overheat Scenario'
    This integration test verifies the critical path of the Coreason Signal architecture:
    1. A Soft Sensor (PINN) detects a critical anomaly (Simulated).
    2. A LogEvent is generated from this anomaly.
    3. The Reflex Engine (RAG) receives the event.
    4. The Engine retrieves the correct SOP from the Vector Store.
    5. The Engine prescribes the correct remediation action (AgentReflex).
    """
    # -------------------------------------------------------------------------
    # PHASE 1: Initialize the Knowledge Base (RAG)
    # -------------------------------------------------------------------------
    # Use a temporary directory for the LanceDB to ensure isolation
    db_path = str(tmp_path / "smoke_test_db")
    vector_store = LocalVectorStore(db_path=db_path)
    # Define the Standard Operating Procedure (SOP) for overheating
    sop_cooling = SOPDocument(
        id="SOP-999",
        title="Critical Reactor Cooling Protocol",
        content="If reactor temperature exceeds 100.0C, immediately engage emergency cooling loop.",
        metadata={"criticality": "high"},
        associated_reflex=AgentReflex(
            action="ENGAGE_COOLING_LOOP",
            parameters={"flow_rate_lpm": 50.0, "override_safety": True},
            reasoning="SOP-999 mandates immediate cooling for temp > 100C",
        ),
    )
    # Ingest SOP into Vector Store
    vector_store.add_sops([sop_cooling])
    # Initialize the Decision Engine
    reflex_engine = ReflexEngine(vector_store=vector_store)
    # -------------------------------------------------------------------------
    # PHASE 2: Simulate Soft Sensor Anomaly (Virtual Sensing)
    # -------------------------------------------------------------------------
    # Configure a virtual sensor for Reactor Temperature
    sensor_config = SoftSensorModel(
        id="v-sensor-reactor-temp",
        input_sensors=["pressure", "agitation_speed"],
        target_variable="reactor_temp_c",
        physics_constraints={"max_temp": "150.0"},
        model_artifact=b"dummy_onnx_bytes",
    )
    sensor_engine = SoftSensorEngine(sensor_config)
    # Mock the ONNX model returning a critical value (e.g., 105.0 degrees)
    # The real model would calculate this from pressure/agitation.
    mock_onnx_session.run.return_value = [np.array([[105.0]], dtype=np.float32)]
    # Run inference
    inputs = {"pressure": 2.5, "agitation_speed": 1200.0}
    prediction = sensor_engine.infer(inputs)
    detected_temp = prediction["reactor_temp_c"]
    assert detected_temp == 105.0, "Soft Sensor failed to predict simulated overheat."
    # -------------------------------------------------------------------------
    # PHASE 3: Trigger the Reflex Loop (The "Signal")
    # -------------------------------------------------------------------------
    # Simulate the application logic: If temp > 100, generate an Error Log
    log_event = LogEvent(
        id="EVT-CRIT-001",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="v-sensor-reactor-temp",
        message=f"CRITICAL: Reactor Temperature {detected_temp}C exceeds limit. Initiate protocol.",
    )
    # -------------------------------------------------------------------------
    # PHASE 4: Autonomous Decision (The "Reflex")
    # -------------------------------------------------------------------------
    # The engine must decide what to do based on the log message
    from coreason_identity.models import UserContext
    from coreason_identity.types import SecretStr

    ctx = UserContext(user_id=SecretStr("sys"), roles=["system"], metadata={})
    reflex = reflex_engine.decide(log_event, ctx)
    # -------------------------------------------------------------------------
    # PHASE 5: Verification
    # -------------------------------------------------------------------------
    assert reflex is not None, "Reflex Engine failed to produce a decision."
    # Check 1: Did we select the correct Action?
    assert reflex.action == "ENGAGE_COOLING_LOOP", (
        f"Incorrect action taken. Expected ENGAGE_COOLING_LOOP, got {reflex.action}"
    )
