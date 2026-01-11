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

from pydantic import BaseModel, Field


class DeviceDefinition(BaseModel):
    """
    Schema for mapping a physical instrument to the system.
    """

    id: str = Field(..., description="Unique identifier for the device, e.g., 'LiquidHandler-01'")
    driver_type: str = Field(..., description="Type of driver to use, e.g., 'SiLA2', 'SerialWrapper', 'VisionWrapper'")
    endpoint: str = Field(..., description="Network endpoint, e.g., 'https://192.168.1.50:50052'")
    capabilities: List[str] = Field(..., description="List of capabilities, e.g., ['Transfer', 'Wash', 'Heater']")
    edge_agent_model: str = Field(..., description="Path or name of the Edge AI model, e.g., 'phi-4-quantized.onnx'")
    allowed_reflexes: List[str] = Field(..., description="List of allowed reflexes, e.g., ['RETRY', 'PAUSE', 'ABORT']")


class SoftSensorModel(BaseModel):
    """
    Schema for Soft Sensor models that infer unmeasurable biological states.
    """

    id: str = Field(..., description="Unique identifier for the model, e.g., 'model_titer_pred_v2'")
    input_sensors: List[str] = Field(..., description="List of input sensors, e.g., ['ph', 'do2', 'agitation']")
    target_variable: str = Field(..., description="The variable being predicted, e.g., 'titer_g_L'")
    physics_constraints: Dict[str, str] = Field(
        ..., description="Physics constraints for the model, e.g., {'min_titer': '0.0'}"
    )
    model_artifact: bytes = Field(..., description="The binary content of the ONNX model artifact")
