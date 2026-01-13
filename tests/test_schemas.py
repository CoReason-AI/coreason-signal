from datetime import datetime

import pytest
from pydantic import ValidationError

from coreason_signal.schemas import AgentReflex, DeviceDefinition, LogEvent, SoftSensorModel


def test_device_definition_valid() -> None:
    """Test creating a valid DeviceDefinition."""
    device = DeviceDefinition(
        id="LiquidHandler-01",
        driver_type="SiLA2",
        endpoint="https://192.168.1.50:50052",
        capabilities=["Transfer", "Wash"],
        edge_agent_model="phi-4.onnx",
        allowed_reflexes=["RETRY", "ABORT"],
    )
    assert device.id == "LiquidHandler-01"
    assert str(device.endpoint) == "https://192.168.1.50:50052/"
    assert device.allowed_reflexes == ["RETRY", "ABORT"]


def test_device_definition_invalid_url() -> None:
    """Test that an invalid URL raises a ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(
            id="Device-Bad-URL",
            driver_type="SiLA2",
            endpoint="not-a-url",
            capabilities=[],
            edge_agent_model="model.onnx",
            allowed_reflexes=[],
        )
    assert "url" in str(excinfo.value).lower()


def test_soft_sensor_model_valid() -> None:
    """Test creating a valid SoftSensorModel."""
    sensor = SoftSensorModel(
        id="model_titer_v2",
        input_sensors=["ph", "do2"],
        target_variable="titer",
        physics_constraints={"min": "0.0", "max": "10.5"},
        model_artifact=b"fake_onnx_bytes",
    )
    assert sensor.id == "model_titer_v2"
    assert sensor.physics_constraints["min"] == "0.0"
    assert sensor.model_artifact == b"fake_onnx_bytes"


def test_soft_sensor_model_invalid_constraint() -> None:
    """Test that non-numeric constraint values raise a ValueError."""
    with pytest.raises(ValidationError) as excinfo:
        SoftSensorModel(
            id="model_bad_constraint",
            input_sensors=["ph"],
            target_variable="titer",
            physics_constraints={"min": "zero"},  # Invalid
            model_artifact=b"bytes",
        )
    # The custom validator raises a ValueError, which Pydantic wraps in a ValidationError
    assert "must be a numeric string" in str(excinfo.value)
    assert "zero" in str(excinfo.value)


def test_log_event_valid() -> None:
    """Test creating a valid LogEvent."""
    event = LogEvent(
        timestamp=datetime.now(),
        source="LiquidHandler-01",
        level="ERROR",
        raw_message="ERR_VACUUM_PRESSURE_LOW",
        metadata={"error_code": "0x4F"},
    )
    assert event.source == "LiquidHandler-01"
    assert event.level == "ERROR"
    assert event.metadata["error_code"] == "0x4F"


def test_agent_reflex_valid() -> None:
    """Test creating a valid AgentReflex."""
    reflex = AgentReflex(
        reflex_id="reflex-123",
        action="RETRY",
        parameters={"speed": 0.5},
        reasoning="SOP-104 matches error.",
        sop_id="SOP-104",
    )
    assert reflex.action == "RETRY"
    assert reflex.parameters["speed"] == 0.5
    assert reflex.sop_id == "SOP-104"
