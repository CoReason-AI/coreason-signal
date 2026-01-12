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

    def test_url_edge_cases(self) -> None:
        """Test various URL formats including IPv6, query params, and auth."""
        # IPv6
        device_ipv6 = DeviceDefinition(
            id="IPv6-Dev",
            driver_type="SiLA2",
            endpoint="http://[::1]:8080",  # type: ignore
            capabilities=[],
            edge_agent_model="model.onnx",
            allowed_reflexes=[],
        )
        assert device_ipv6.endpoint.host == "[::1]"

        # Query Params
        device_query = DeviceDefinition(
            id="Query-Dev",
            driver_type="SiLA2",
            endpoint="https://api.lab.com/v1/device?token=abc",  # type: ignore
            capabilities=[],
            edge_agent_model="model.onnx",
            allowed_reflexes=[],
        )
        assert device_query.endpoint.query == "token=abc"

        # Auth
        device_auth = DeviceDefinition(
            id="Auth-Dev",
            driver_type="SiLA2",
            endpoint="https://user:pass@192.168.1.50",  # type: ignore
            capabilities=[],
            edge_agent_model="model.onnx",
            allowed_reflexes=[],
        )
        assert device_auth.endpoint.username == "user"
        assert device_auth.endpoint.password == "pass"

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

    def test_onnx_case_sensitivity(self) -> None:
        """Test that .ONNX is rejected (strict lowercase enforcement)."""
        with pytest.raises(ValidationError) as excinfo:
            DeviceDefinition(
                id="Device-Case",
                driver_type="SiLA2",
                endpoint="https://localhost:8080",  # type: ignore
                capabilities=[],
                edge_agent_model="model.ONNX",
                allowed_reflexes=[],
            )
        assert "Edge agent model must be an .onnx file" in str(excinfo.value)

    def test_serialization_roundtrip(self) -> None:
        """Test full JSON serialization and deserialization roundtrip."""
        device = DeviceDefinition(
            id="Complex-Dev",
            driver_type="SiLA2",
            endpoint="https://192.168.1.1:5000/path",  # type: ignore
            capabilities=["A", "B"],
            edge_agent_model="agent.onnx",
            allowed_reflexes=["STOP"],
        )

        json_str = device.model_dump_json()
        device_restored = DeviceDefinition.model_validate_json(json_str)

        assert device_restored.id == device.id
        assert device_restored.endpoint == device.endpoint
        assert device_restored.edge_agent_model == device.edge_agent_model


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

    def test_numeric_constraint_edge_cases(self) -> None:
        """Test scientific notation, negative numbers, and infinity."""
        model = SoftSensorModel(
            id="model_edge",
            input_sensors=[],
            target_variable="y",
            physics_constraints={"sci_not": "1.5e-3", "negative": "-10.5", "infinity": "inf", "neg_infinity": "-inf"},
            model_artifact=b"",
        )
        # Validation passes if no exception is raised
        assert model.physics_constraints["sci_not"] == "1.5e-3"
        # Verify they are actually parseable as floats
        assert float(model.physics_constraints["sci_not"]) == 0.0015
        assert float(model.physics_constraints["infinity"]) == float("inf")

    def test_missing_fields(self) -> None:
        """Test validation for missing required fields."""
        with pytest.raises(ValidationError):
            SoftSensorModel(
                id="model_fail",
                input_sensors=[],
                # Missing target_variable, physics_constraints, model_artifact
            )  # type: ignore

    def test_serialization_roundtrip_with_bytes(self) -> None:
        """Test JSON serialization with binary artifact."""
        model = SoftSensorModel(
            id="model_bytes",
            input_sensors=["a"],
            target_variable="b",
            physics_constraints={"val": "1"},
            model_artifact=b"\xde\xad\xbe\xef",
        )

        json_str = model.model_dump_json()
        model_restored = SoftSensorModel.model_validate_json(json_str)

        assert model_restored.id == model.id
        assert model_restored.model_artifact == model.model_artifact
        assert model_restored.model_artifact == b"\xde\xad\xbe\xef"

    def test_invalid_base64_artifact(self) -> None:
        """Test validation fails for invalid base64 strings."""
        invalid_json = """
        {
            "id": "bad_bytes",
            "input_sensors": [],
            "target_variable": "y",
            "physics_constraints": {},
            "model_artifact": "NOT_BASE64!!!"
        }
        """
        with pytest.raises(ValidationError) as excinfo:
            SoftSensorModel.model_validate_json(invalid_json)
        assert "Invalid base64 encoded string" in str(excinfo.value)

    def test_invalid_type_artifact(self) -> None:
        """Test validation fails for non-bytes/non-string artifacts."""
        with pytest.raises(ValidationError) as excinfo:
            SoftSensorModel(
                id="model_bad_type",
                input_sensors=[],
                target_variable="y",
                physics_constraints={},
                model_artifact=12345,  # type: ignore
            )
        assert "model_artifact must be bytes or a base64 encoded string" in str(excinfo.value)
