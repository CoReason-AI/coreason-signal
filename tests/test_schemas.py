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

from coreason_signal.schemas import DeviceDefinition, SoftSensorModel


def test_device_definition_valid() -> None:
    """Test creating a valid DeviceDefinition."""
    data: Dict[str, Any] = {
        "id": "LiquidHandler-01",
        "driver_type": "SiLA2",
        "endpoint": "https://192.168.1.50:50052",
        "capabilities": ["Transfer", "Wash"],
        "edge_agent_model": "phi-4-quantized.onnx",
        "allowed_reflexes": ["RETRY", "PAUSE"],
    }
    device = DeviceDefinition(**data)
    assert device.id == "LiquidHandler-01"
    assert device.driver_type == "SiLA2"
    assert str(device.endpoint) == "https://192.168.1.50:50052/"
    assert len(device.capabilities) == 2


def test_device_definition_missing_field() -> None:
    """Test that missing required fields raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "LiquidHandler-01",
        # Missing driver_type
        "endpoint": "https://192.168.1.50:50052",
        "capabilities": ["Transfer"],
        "edge_agent_model": "phi-4-quantized.onnx",
        "allowed_reflexes": ["RETRY"],
    }
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(**data)
    assert "driver_type" in str(excinfo.value)


def test_device_definition_invalid_driver_type() -> None:
    """Test that an invalid driver_type raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "LiquidHandler-01",
        "driver_type": "InvalidDriver",
        "endpoint": "https://192.168.1.50:50052",
        "capabilities": ["Transfer"],
        "edge_agent_model": "phi-4-quantized.onnx",
        "allowed_reflexes": ["RETRY"],
    }
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(**data)
    assert "Input should be 'SiLA2', 'SerialWrapper' or 'VisionWrapper'" in str(excinfo.value)


def test_device_definition_invalid_endpoint_url() -> None:
    """Test that an invalid endpoint URL raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "LiquidHandler-01",
        "driver_type": "SiLA2",
        "endpoint": "not-a-url",
        "capabilities": ["Transfer"],
        "edge_agent_model": "phi-4-quantized.onnx",
        "allowed_reflexes": ["RETRY"],
    }
    with pytest.raises(ValidationError):
        DeviceDefinition(**data)


def test_device_definition_empty_id() -> None:
    """Test that empty id raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "",
        "driver_type": "SiLA2",
        "endpoint": "https://localhost",
        "capabilities": ["Transfer"],
        "edge_agent_model": "model.onnx",
        "allowed_reflexes": [],
    }
    with pytest.raises(ValidationError):
        DeviceDefinition(**data)


def test_device_definition_empty_capabilities() -> None:
    """Test that empty capabilities list raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "Device1",
        "driver_type": "SiLA2",
        "endpoint": "https://localhost",
        "capabilities": [],  # Empty
        "edge_agent_model": "model.onnx",
        "allowed_reflexes": [],
    }
    with pytest.raises(ValidationError):
        DeviceDefinition(**data)


def test_device_definition_extra_fields() -> None:
    """Test that extra fields are forbidden."""
    data: Dict[str, Any] = {
        "id": "Device1",
        "driver_type": "SiLA2",
        "endpoint": "https://localhost",
        "capabilities": ["Transfer"],
        "edge_agent_model": "model.onnx",
        "allowed_reflexes": [],
        "extra_field": "not_allowed",
    }
    with pytest.raises(ValidationError) as excinfo:
        DeviceDefinition(**data)
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_soft_sensor_model_valid() -> None:
    """Test creating a valid SoftSensorModel."""
    data: Dict[str, Any] = {
        "id": "model_titer_pred_v2",
        "input_sensors": ["ph", "do2"],
        "target_variable": "titer_g_L",
        "physics_constraints": {"min_titer": "0.0"},
        "model_artifact": b"fake_onnx_bytes",
    }
    sensor = SoftSensorModel(**data)
    assert sensor.id == "model_titer_pred_v2"
    assert sensor.target_variable == "titer_g_L"
    assert sensor.model_artifact == b"fake_onnx_bytes"


def test_soft_sensor_model_invalid_types() -> None:
    """Test that invalid types raise ValidationError."""
    data: Dict[str, Any] = {
        "id": "model_titer_pred_v2",
        "input_sensors": "not_a_list",  # Invalid type
        "target_variable": "titer_g_L",
        "physics_constraints": {"min_titer": "0.0"},
        "model_artifact": b"fake_onnx_bytes",
    }
    with pytest.raises(ValidationError):
        SoftSensorModel(**data)


def test_soft_sensor_model_empty_artifact() -> None:
    """Test that empty model artifact raises ValidationError."""
    data: Dict[str, Any] = {
        "id": "model1",
        "input_sensors": ["s1"],
        "target_variable": "target",
        "physics_constraints": {},
        "model_artifact": b"",  # Empty
    }
    with pytest.raises(ValidationError):
        SoftSensorModel(**data)


def test_soft_sensor_model_extra_fields() -> None:
    """Test that extra fields are forbidden."""
    data: Dict[str, Any] = {
        "id": "model1",
        "input_sensors": ["s1"],
        "target_variable": "target",
        "physics_constraints": {},
        "model_artifact": b"123",
        "extra_field": 123,
    }
    with pytest.raises(ValidationError):
        SoftSensorModel(**data)
