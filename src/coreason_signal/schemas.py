# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import base64
from typing import Any, Dict, List

from pydantic import BaseModel, HttpUrl, field_serializer, field_validator


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

    @field_serializer("model_artifact")
    def serialize_model_artifact(self, model_artifact: bytes, _info: Any) -> str:
        """Serializes the bytes artifact to a base64 string for JSON compatibility."""
        return base64.b64encode(model_artifact).decode("utf-8")

    @field_validator("model_artifact", mode="before")
    @classmethod
    def validate_model_artifact(cls, v: Any) -> bytes:
        """Decodes the base64 string back to bytes if the input is a string."""
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception as e:
                raise ValueError("Invalid base64 encoded string for model_artifact") from e
        if isinstance(v, bytes):
            return v
        raise ValueError("model_artifact must be bytes or a base64 encoded string")
