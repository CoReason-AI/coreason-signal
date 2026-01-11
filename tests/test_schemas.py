# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from datetime import datetime
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

# --- DeviceDefinition Tests ---


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
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(
            id="LiquidHandler-01",
            driver_type="SiLA2",
            endpoint="not-a-url",  # type: ignore[arg-type]
            capabilities=["Transfer"],
            edge_agent_model="phi-4.onnx",
            allowed_reflexes=["RETRY"],
        )
    assert "url" in str(excinfo.value)


def test_device_definition_empty_capabilities() -> None:
    """Edge Case: Empty capabilities list should fail (min_length=1)."""
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(
            id="Dev1",
            driver_type="SiLA2",
            endpoint="http://localhost",  # type: ignore[arg-type]
            capabilities=[],  # Empty list
            edge_agent_model="model",
            allowed_reflexes=[],
        )
    # Check for specific error location or message part if possible
    assert "List should have at least 1 item" in str(excinfo.value)


def test_device_definition_invalid_driver_enum() -> None:
    """Edge Case: Invalid Literal for driver_type."""
    with pytest.raises(ValidationError):
        DeviceDefinition(
            id="Dev1",
            driver_type="UnknownDriver",  # type: ignore[arg-type]
            endpoint="http://localhost",  # type: ignore[arg-type]
            capabilities=["A"],
            edge_agent_model="model",
            allowed_reflexes=[],
        )


# --- SoftSensorModel Tests ---


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


def test_soft_sensor_model_empty_constraints() -> None:
    """Complex Case: Physics constraints can be empty if not strictly required."""
    model = SoftSensorModel(
        id="model_v1",
        input_sensors=["ph"],
        target_variable="yield",
        physics_constraints={},
        model_artifact=b"data",
    )
    assert model.physics_constraints == {}


def test_soft_sensor_model_strict_types() -> None:
    """Edge Case: Verify physics_constraints enforces strings (strict types)."""
    with pytest.raises(ValidationError):
        SoftSensorModel(
            id="model_v1",
            input_sensors=["ph"],
            target_variable="yield",
            # Pass int '0', expect failure (no coercion)
            physics_constraints={"min": 0},  # type: ignore[dict-item]
            model_artifact=b"data",
        )


# --- SOPDocument Tests ---


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


def test_sop_document_empty_fields() -> None:
    """Edge Case: Empty string for required fields with min_length=1."""
    with pytest.raises(ValidationError):
        SOPDocument(
            id="",
            title="Title",
            content="Content",
        )
    with pytest.raises(ValidationError):
        SOPDocument(
            id="ID",
            title="",
            content="Content",
        )


def test_sop_document_unicode() -> None:
    """Complex Case: Unicode characters in ID and content."""
    doc = SOPDocument(
        id="SOP-Ω",
        title="Temperature Check °C",
        content="Ensure temp is < 100°C. 警告.",
    )
    assert doc.id == "SOP-Ω"
    assert "警告" in doc.content


def test_sop_document_extra_fields() -> None:
    """Edge Case: Extra fields should be forbidden (ConfigDict extra='forbid')."""
    with pytest.raises(ValidationError) as excinfo:
        SOPDocument(
            id="SOP-1",
            title="T",
            content="C",
            random_field="Not allowed",  # type: ignore[call-arg]
        )
    assert "Extra inputs are not permitted" in str(excinfo.value)


# --- LogEvent Tests ---


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


def test_log_event_timestamp_parsing() -> None:
    """Complex Case: Parse timestamp from ISO string."""
    ts_str = "2023-10-01T12:00:00Z"
    event = LogEvent(
        source_device_id="Dev1",
        message="Msg",
        level="INFO",
        timestamp=ts_str,  # type: ignore[arg-type]
    )
    assert event.timestamp.year == 2023
    assert event.timestamp.month == 10
    # Pydantic preserves timezone info if provided
    assert event.timestamp.tzinfo is not None


def test_log_event_invalid_level() -> None:
    with pytest.raises(ValidationError):
        LogEvent(
            source_device_id="Dev1",
            message="Msg",
            level="CRITICAL_FAIL",  # type: ignore[arg-type]
        )


def test_log_event_context_data_mixed_types() -> None:
    """Edge Case: Verify context_data accepts mixed types."""
    event = LogEvent(
        source_device_id="Dev1",
        message="Msg",
        level="INFO",
        context_data={"code": 404, "retries": 3, "valid": True, "temp": 37.5},
    )
    assert event.context_data["code"] == 404
    assert event.context_data["retries"] == 3
    assert event.context_data["valid"] is True
    assert event.context_data["temp"] == 37.5


# --- AgentReflex Tests ---


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


def test_agent_reflex_mixed_parameters() -> None:
    """Complex Case: Parameters dict allowing str, int, float."""
    reflex = AgentReflex(
        id="act-1",
        action_type="NOTIFY",
        reasoning="Testing params",
        parameters={
            "str_param": "hello",
            "int_param": 10,
            "float_param": 3.14,
        },
    )
    assert reflex.parameters["str_param"] == "hello"
    assert reflex.parameters["int_param"] == 10
    assert reflex.parameters["float_param"] == 3.14


def test_agent_reflex_timestamp_auto() -> None:
    """Edge Case: Ensure default factory works and creates unique times (mostly)."""
    r1 = AgentReflex(id="1", action_type="IGNORE", reasoning="r")
    r2 = AgentReflex(id="2", action_type="IGNORE", reasoning="r")
    # They might be equal if executed extremely fast, but they should be valid datetimes
    assert isinstance(r1.timestamp, datetime)
    assert isinstance(r2.timestamp, datetime)
