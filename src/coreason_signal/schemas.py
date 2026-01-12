# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Dict, List

from pydantic import BaseModel, HttpUrl, field_validator


class DeviceDefinition(BaseModel):
    """
    Defines the configuration and capabilities of a laboratory instrument.

    Attributes:
        id: Unique identifier for the device (e.g., "LiquidHandler-01").
        driver_type: The type of driver interface ("SiLA2", "SerialWrapper", "VisionWrapper").
        endpoint: The network address of the device (must be a valid URL).
        capabilities: List of capabilities the device supports.
        edge_agent_model: Filename of the ONNX model for the edge agent.
        allowed_reflexes: List of autonomous actions the agent is allowed to trigger.
    """

    id: str
    driver_type: str
    endpoint: HttpUrl
    capabilities: List[str]
    edge_agent_model: str
    allowed_reflexes: List[str]

    @field_validator("edge_agent_model")
    @classmethod
    def validate_onnx_extension(cls, v: str) -> str:
        """Validates that the model file has a .onnx extension."""
        if not v.endswith(".onnx"):
            raise ValueError("Edge agent model must be an .onnx file")
        return v


class SoftSensorModel(BaseModel):
    """
    Defines the configuration for a Soft Sensor (Virtual Metrology).

    Attributes:
        id: Unique identifier for the model.
        input_sensors: List of physical sensor inputs required (e.g., ["ph", "do2"]).
        target_variable: The output variable being predicted (e.g., "titer_g_L").
        physics_constraints: Dictionary of physical constraints (e.g., {"min_titer": "0.0"}).
        model_artifact: The binary content of the ONNX model file.
    """

    id: str
    input_sensors: List[str]
    target_variable: str
    physics_constraints: Dict[str, str]
    model_artifact: bytes

    @field_validator("physics_constraints")
    @classmethod
    def validate_constraints_are_numeric(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validates that constraint values are parsable as numbers."""
        for key, value in v.items():
            try:
                float(value)
            except ValueError as e:
                raise ValueError(f"Constraint value for '{key}' must be numeric, got '{value}'") from e
        return v
