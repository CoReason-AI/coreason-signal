# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

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


class SOPDocument(BaseModel):
    """
    Schema for a Standard Operating Procedure (SOP) used in the Edge Agent's RAG system.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique SOP identifier, e.g., 'SOP-104'")
    title: str = Field(..., min_length=1, description="Title of the SOP")
    content: str = Field(..., min_length=1, description="Full text content of the SOP")
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata keys, e.g., {'category': 'maintenance'}"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Pre-computed vector embedding of the content (optional)"
    )


class LogEvent(BaseModel):
    """
    Schema for an incoming log or telemetry event from a device.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of the event"
    )
    source_device_id: str = Field(..., min_length=1, description="ID of the device generating the log")
    error_code: Optional[str] = Field(default=None, description="Vendor-specific error code, e.g., 'ERR_VACUUM_LOW'")
    message: str = Field(..., min_length=1, description="Human-readable log message")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(..., description="Log severity level")
    context_data: Dict[str, str] = Field(
        default_factory=dict, description="Contextual data, e.g., {'speed': '100', 'tip_pos': 'A1'}"
    )


class AgentReflex(BaseModel):
    """
    Schema for the decision or action taken by the Edge Agent.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique ID for this reflex action event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of the decision"
    )
    related_log_id: Optional[str] = Field(default=None, description="ID of the log event that triggered this reflex")
    action_type: Literal["RETRY", "PAUSE", "ABORT", "NOTIFY", "IGNORE"] = Field(
        ..., description="The type of action taken"
    )
    reasoning: str = Field(..., min_length=1, description="Explanation for the decision (from LLM or RAG)")
    parameters: Dict[str, float | str | int] = Field(
        default_factory=dict, description="Parameters for the action, e.g., {'speed_factor': 0.5}"
    )
