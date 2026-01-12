import pytest
from pydantic import ValidationError

from coreason_signal.schemas import DeviceDefinition, SoftSensorModel


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
