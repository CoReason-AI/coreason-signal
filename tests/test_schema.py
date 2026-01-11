# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import pytest
from pydantic import ValidationError

from coreason_signal.schema import DeviceDefinition, SoftSensorModel


class TestDeviceDefinition:
    def test_valid_instantiation(self) -> None:
        """Test creating a valid DeviceDefinition."""
        device = DeviceDefinition(
            id="LiquidHandler-01",
            driver_type="SiLA2",
            endpoint="https://192.168.1.50:50052",
            capabilities=["Transfer", "Wash"],
            edge_agent_model="phi-4-quantized.onnx",
            allowed_reflexes=["RETRY", "PAUSE"],
        )
        assert device.id == "LiquidHandler-01"
        assert device.driver_type == "SiLA2"
        assert device.endpoint == "https://192.168.1.50:50052"
        assert device.capabilities == ["Transfer", "Wash"]
        assert device.edge_agent_model == "phi-4-quantized.onnx"
        assert device.allowed_reflexes == ["RETRY", "PAUSE"]

    def test_invalid_driver_type(self) -> None:
        """Test that invalid driver_type raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            DeviceDefinition(
                id="LiquidHandler-01",
                driver_type="InvalidDriver",
                endpoint="https://localhost:5000",
                capabilities=[],
                edge_agent_model="model.onnx",
                allowed_reflexes=[],
            )
        assert "Input should be 'SiLA2', 'SerialWrapper' or 'VisionWrapper'" in str(excinfo.value)

    def test_missing_fields(self) -> None:
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            DeviceDefinition(id="IncompleteDevice")  # type: ignore


class TestSoftSensorModel:
    def test_valid_instantiation(self) -> None:
        """Test creating a valid SoftSensorModel."""
        model = SoftSensorModel(
            id="model_titer_pred_v2",
            input_sensors=["ph", "do2", "agitation"],
            target_variable="titer_g_L",
            physics_constraints={"min_titer": "0.0"},
            model_artifact=b"\x00\x01\x02",
        )
        assert model.id == "model_titer_pred_v2"
        assert model.input_sensors == ["ph", "do2", "agitation"]
        assert model.target_variable == "titer_g_L"
        assert model.physics_constraints == {"min_titer": "0.0"}
        assert model.model_artifact == b"\x00\x01\x02"

    def test_invalid_types(self) -> None:
        """Test type validation for fields."""
        with pytest.raises(ValidationError):
            SoftSensorModel(
                id="model_bad",
                input_sensors="not_a_list",
                target_variable="titer",
                physics_constraints={},
                model_artifact=b"",
            )
