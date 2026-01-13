# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Any, Dict, List, Literal, Optional

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


class DeviceDefinition(BaseModel):
    """
    Schema for mapping a physical instrument to the system.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique identifier for the device, e.g., 'LiquidHandler-01'")
    driver_type: Literal["SiLA2", "SerialWrapper", "VisionWrapper"] = Field(..., description="Type of driver to use.")
    endpoint: AnyUrl = Field(..., description="Network endpoint, e.g., 'https://192.168.1.50:50052'")
    capabilities: List[str] = Field(
        ..., min_length=1, description="List of capabilities, e.g., ['Transfer', 'Wash', 'Heater']"
    )
    edge_agent_model: str = Field(
        ..., min_length=1, description="Path or name of the Edge AI model, e.g., 'phi-4-quantized.onnx'"
    )
    allowed_reflexes: List[str] = Field(..., description="List of allowed reflexes, e.g., ['RETRY', 'PAUSE', 'ABORT']")


class SoftSensorModel(BaseModel):
    """
    Schema for Soft Sensor models that infer unmeasurable biological states.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique identifier for the model, e.g., 'model_titer_pred_v2'")
    input_sensors: List[str] = Field(
        ..., min_length=1, description="List of input sensors, e.g., ['ph', 'do2', 'agitation']"
    )
    target_variable: str = Field(..., min_length=1, description="The variable being predicted, e.g., 'titer_g_L'")
    physics_constraints: Dict[str, str] = Field(
        ..., description="Physics constraints for the model, e.g., {'min_titer': '0.0'}"
    )
    model_artifact: bytes = Field(..., min_length=1, description="The binary content of the ONNX model artifact")


class AgentReflex(BaseModel):
    """
    Schema for an autonomous action taken by the Edge Agent.
    """

    model_config = ConfigDict(extra="forbid")

    action_name: str = Field(..., min_length=1, description="Name of the action, e.g., 'Aspirate'")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action, e.g., {'speed': 0.5}"
    )
    reasoning: str = Field(..., description="Explanation for why this reflex was triggered.")


class SOPDocument(BaseModel):
    """
    Schema for a Standard Operating Procedure (SOP) document used for RAG.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=1, description="Unique identifier for the SOP, e.g., 'SOP-104'")
    title: str = Field(..., min_length=1, description="Title of the SOP")
    content: str = Field(..., min_length=1, description="Text content to be embedded for retrieval.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
    associated_reflex: Optional[AgentReflex] = Field(
        None, description="The reflex action prescribed by this SOP, if any."
    )


class LogEvent(BaseModel):
    """
    Schema for a log event that triggers the Edge Agent.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique Event ID")
    timestamp: str = Field(..., description="ISO 8601 Timestamp")
    message: str = Field(..., description="The semantic log message, e.g., 'Vacuum Pressure Low'")
    raw_code: Optional[str] = Field(None, description="Original error code, e.g., 'ERR_0x4F'")
