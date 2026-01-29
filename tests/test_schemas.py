import datetime

import pytest
from coreason_signal.schemas import AgentReflex, DeviceDefinition, LogEvent, SoftSensorModel
from pydantic import ValidationError


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
        physics_constraints={"min": 0.0, "max": 10.5},
        model_artifact=b"fake_onnx_bytes",
    )
    assert sensor.id == "model_titer_v2"
    assert sensor.physics_constraints["min"] == 0.0
    assert sensor.model_artifact == b"fake_onnx_bytes"


def test_soft_sensor_model_invalid_constraint() -> None:
    """Test that non-numeric constraint values raise a ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        SoftSensorModel(
            id="model_bad_constraint",
            input_sensors=["ph"],
            target_variable="titer",
            physics_constraints={"min": "zero"},  # Invalid: cannot parse to float
            model_artifact=b"bytes",
        )
    # Pydantic raises a validation error for float parsing
    assert "Input should be a valid number" in str(excinfo.value)


def test_log_event_valid() -> None:
    """Test creating a valid LogEvent."""
    event = LogEvent(
        id="evt-001",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="LiquidHandler-01",
        message="ERR_VACUUM_PRESSURE_LOW",
        raw_code="0x4F",
    )
    assert event.id == "evt-001"
    assert event.level == "ERROR"
    assert event.source == "LiquidHandler-01"
    assert event.message == "ERR_VACUUM_PRESSURE_LOW"
    assert event.raw_code == "0x4F"


def test_agent_reflex_valid() -> None:
    """Test creating a valid AgentReflex."""
    reflex = AgentReflex(
        action="RETRY",
        parameters={"speed": 0.5},
        reasoning="SOP-104 matches error.",
    )
    assert reflex.action == "RETRY"
    assert reflex.parameters["speed"] == 0.5
    assert reflex.reasoning == "SOP-104 matches error."
