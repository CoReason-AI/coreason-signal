# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class DeviceDefinition(BaseModel):
    """
    Configuration for a hardware device connected to the signal gateway.
    Maps physical instruments to their digital drivers and capabilities.
    """

    id: str = Field(..., description="Unique identifier for the device, e.g., 'LiquidHandler-01'")
    driver_type: Literal["SiLA2", "SerialWrapper", "VisionWrapper"] = Field(
        ..., description="The type of driver interface used to communicate with the device."
    )
    endpoint: str = Field(..., description="Connection endpoint, e.g., 'https://192.168.1.50:50052'")
    capabilities: List[str] = Field(..., description="List of supported capabilities, e.g., ['Transfer', 'Wash']")

    # Edge AI Config
    edge_agent_model: str = Field(
        ..., description="Filename or ID of the quantized Micro-LLM model, e.g., 'phi-4-quantized.onnx'"
    )
    allowed_reflexes: List[str] = Field(
        ..., description="List of allowed autonomous actions, e.g., ['RETRY', 'PAUSE', 'ABORT']"
    )


class SoftSensorModel(BaseModel):
    """
    Definition for a Soft Sensor (Virtual Metrology) model.
    """

    id: str = Field(..., description="Unique identifier for the model, e.g., 'model_titer_pred_v2'")
    input_sensors: List[str] = Field(..., description="List of input signal keys, e.g., ['ph', 'do2', 'agitation']")
    target_variable: str = Field(..., description="The variable being inferred, e.g., 'titer_g_L'")
    physics_constraints: Dict[str, str] = Field(
        ..., description="Physics-based constraints for the model, e.g., {'min_titer': '0.0'}"
    )
    model_artifact: bytes = Field(..., description="The serialized ONNX model artifact")
