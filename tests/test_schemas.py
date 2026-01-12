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

from coreason_signal.schemas import DeviceDefinition, SoftSensorModel


class TestDeviceDefinition:
    def test_valid_device_definition(self) -> None:
        """Test creating a valid DeviceDefinition."""
        device = DeviceDefinition(
            id="LiquidHandler-01",
            driver_type="SiLA2",
            endpoint="https://192.168.1.50:50052",  # type: ignore
            capabilities=["Transfer", "Wash"],
            edge_agent_model="phi-4-quantized.onnx",
            allowed_reflexes=["RETRY", "ABORT"],
        )
        assert device.id == "LiquidHandler-01"
        assert str(device.endpoint) == "https://192.168.1.50:50052/"
        assert device.edge_agent_model == "phi-4-quantized.onnx"

    def test_invalid_url(self) -> None:
        """Test that invalid URLs raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            DeviceDefinition(
                id="Device-01",
                driver_type="SiLA2",
                endpoint="not-a-url",  # type: ignore
                capabilities=[],
                edge_agent_model="model.onnx",
                allowed_reflexes=[],
            )
        assert "Input should be a valid URL" in str(excinfo.value) or "url" in str(excinfo.value)

    def test_invalid_onnx_extension(self) -> None:
        """Test that edge_agent_model must end with .onnx."""
        with pytest.raises(ValidationError) as excinfo:
            DeviceDefinition(
                id="Device-01",
                driver_type="SiLA2",
                endpoint="https://localhost:8080",  # type: ignore
                capabilities=[],
                edge_agent_model="model.pt",
                allowed_reflexes=[],
            )
        assert "Edge agent model must be an .onnx file" in str(excinfo.value)


class TestSoftSensorModel:
    def test_valid_soft_sensor_model(self) -> None:
        """Test creating a valid SoftSensorModel."""
        model = SoftSensorModel(
            id="model_titer_pred_v2",
            input_sensors=["ph", "do2"],
            target_variable="titer_g_L",
            physics_constraints={"min_titer": "0.0", "max_titer": "10.5"},
            model_artifact=b"fake-onnx-bytes",
        )
        assert model.id == "model_titer_pred_v2"
        assert model.physics_constraints["min_titer"] == "0.0"

    def test_invalid_numeric_constraint(self) -> None:
        """Test that physics_constraints values must be numeric strings."""
        with pytest.raises(ValidationError) as excinfo:
            SoftSensorModel(
                id="model_fail",
                input_sensors=[],
                target_variable="y",
                physics_constraints={"min_val": "five"},
                model_artifact=b"",
            )
        assert "Constraint value for 'min_val' must be numeric, got 'five'" in str(excinfo.value)

    def test_missing_fields(self) -> None:
        """Test validation for missing required fields."""
        with pytest.raises(ValidationError):
            SoftSensorModel(
                id="model_fail",
                input_sensors=[],
                # Missing target_variable, physics_constraints, model_artifact
            )  # type: ignore
