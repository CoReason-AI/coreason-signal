# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from coreason_signal.schemas import (
    AgentReflex,
    DeviceDefinition,
    LogEvent,
    SoftSensorModel,
    SOPDocument,
)


def test_device_definition_valid() -> None:
    model = DeviceDefinition(
        id="LiquidHandler-01",
        driver_type="SiLA2",
        endpoint="https://192.168.1.50:50052",  # type: ignore[arg-type]
        capabilities=["Transfer", "Wash"],
        edge_agent_model="phi-4.onnx",
        allowed_reflexes=["RETRY", "ABORT"],
    )
    assert model.id == "LiquidHandler-01"
    assert str(model.endpoint) == "https://192.168.1.50:50052/"


def test_device_definition_invalid_url() -> None:
    with pytest.raises(ValidationError):
        DeviceDefinition(
            id="LiquidHandler-01",
            driver_type="SiLA2",
            endpoint="not-a-url",  # type: ignore[arg-type]
            capabilities=["Transfer"],
            edge_agent_model="phi-4.onnx",
            allowed_reflexes=["RETRY"],
        )


def test_soft_sensor_model_valid() -> None:
    model = SoftSensorModel(
        id="model_v1",
        input_sensors=["ph", "temp"],
        target_variable="yield",
        physics_constraints={"min": "0"},
        model_artifact=b"binarydata",
    )
    assert model.id == "model_v1"
    assert model.model_artifact == b"binarydata"


def test_sop_document_valid() -> None:
    doc = SOPDocument(
        id="SOP-104",
        title="Vacuum Error Handling",
        content="If vacuum pressure is low, retry at 50% speed.",
        metadata={"category": "maintenance"},
        embedding=[0.1, 0.2, 0.3],
    )
    assert doc.id == "SOP-104"
    assert doc.embedding == [0.1, 0.2, 0.3]


def test_sop_document_minimal() -> None:
    doc = SOPDocument(
        id="SOP-105",
        title="Minimal SOP",
        content="Just content.",
    )
    assert doc.id == "SOP-105"
    assert doc.embedding is None
    assert doc.metadata == {}


def test_log_event_valid() -> None:
    event = LogEvent(
        source_device_id="BioReactor-01",
        error_code="ERR_TEMP_HIGH",
        message="Temperature exceeded limit",
        level="ERROR",
        context_data={"temp": "38.5"},
    )
    assert event.source_device_id == "BioReactor-01"
    assert event.timestamp is not None
    assert event.level == "ERROR"


def test_log_event_invalid_level() -> None:
    # Casting to Any to trick mypy into allowing the bad call during test
    bad_data: Dict[str, Any] = {
        "source_device_id": "Dev1",
        "message": "Info",
        "level": "INVALID_LEVEL",
    }
    with pytest.raises(ValidationError):
        LogEvent(**bad_data)


def test_agent_reflex_valid() -> None:
    reflex = AgentReflex(
        id="action-123",
        related_log_id="log-456",
        action_type="RETRY",
        reasoning="SOP-104 suggests retry.",
        parameters={"speed_factor": 0.5},
    )
    assert reflex.action_type == "RETRY"
    assert reflex.parameters["speed_factor"] == 0.5
    assert reflex.timestamp is not None


def test_agent_reflex_missing_fields() -> None:
    bad_data: Dict[str, Any] = {
        "id": "action-123",
        # Missing action_type
        "reasoning": "Because I said so",
    }
    with pytest.raises(ValidationError):
        AgentReflex(**bad_data)
