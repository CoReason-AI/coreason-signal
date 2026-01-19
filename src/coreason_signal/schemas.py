# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DeviceDefinition(BaseModel):
    """Hardware abstraction layer mapping for SiLA 2 and legacy instruments.

    Attributes:
        id (str): Unique identifier for the device (e.g., "LiquidHandler-01").
        driver_type (str): Type of driver to use (e.g., "SiLA2", "SerialWrapper").
        endpoint (HttpUrl): Network endpoint of the device (e.g., "https://192.168.1.50:50052").
        capabilities (List[str]): List of capabilities supported by the device.
        edge_agent_model (str): Identifier for the edge agent model to use.
        allowed_reflexes (List[str]): List of allowed reflex actions (e.g., ["RETRY", "PAUSE"]).
    """

    id: str
    driver_type: str
    endpoint: HttpUrl
    capabilities: List[str]

    # Edge AI Config
    edge_agent_model: str
    allowed_reflexes: List[str]


class SoftSensorModel(BaseModel):
    """Configuration for physics-informed neural networks (PINNs) acting as virtual sensors.

    Attributes:
        id (str): Unique identifier for the sensor model.
        input_sensors (List[str]): List of input sensor keys required by the model.
        target_variable (str): The name of the output variable being predicted.
        physics_constraints (Dict[str, float]): Constraints to apply to the output (e.g., {"min": 0.0}).
        model_artifact (bytes): The binary content of the ONNX model file.
    """

    id: str
    input_sensors: List[str]
    target_variable: str
    physics_constraints: Dict[str, float]
    model_artifact: bytes


class AgentReflex(BaseModel):
    """Schema for an autonomous action taken by the Edge Agent.

    Attributes:
        action (str): The name of the action to take (e.g., "RETRY", "PAUSE").
        parameters (Dict[str, Any]): Additional parameters for the action.
        reasoning (str): Textual explanation for why this action was chosen.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., min_length=1, description="Name of the action, e.g., 'Aspirate'")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action, e.g., {'speed': 0.5}"
    )
    reasoning: str = Field(..., description="Explanation for why this reflex was triggered.")


class SOPDocument(BaseModel):
    """Schema for a Standard Operating Procedure (SOP) document used for RAG.

    Attributes:
        id (str): Unique identifier for the SOP.
        title (str): Title of the SOP.
        content (str): The text content used for semantic search.
        metadata (Dict[str, Any]): Arbitrary metadata associated with the SOP.
        associated_reflex (Optional[AgentReflex]): A pre-defined reflex action for this SOP.
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
    """Schema for a log event that triggers the Edge Agent.

    Attributes:
        id (str): Unique Event ID.
        timestamp (str): ISO 8601 Timestamp of the event.
        level (str): Log severity level (e.g., "INFO", "ERROR").
        source (str): Identifier of the source component/instrument.
        message (str): The main log message.
        raw_code (Optional[str]): Original error code from the hardware, if available.
        metadata (Dict[str, Any]): Additional context for the event.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique Event ID")
    timestamp: str = Field(..., description="ISO 8601 Timestamp")
    level: str = Field(..., description="Log level, e.g., 'INFO', 'ERROR'")
    source: str = Field(..., description="Source component/instrument ID")
    message: str = Field(..., description="The semantic log message, e.g., 'Vacuum Pressure Low'")
    raw_code: Optional[str] = Field(None, description="Original error code, e.g., 'ERR_0x4F'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SemanticFact(BaseModel):
    """Represents a derived semantic fact (Subject-Predicate-Object).

    Used for constructing the Knowledge Graph.

    Attributes:
        subject (str): The subject node ID (e.g., 'Bioreactor-01').
        predicate (str): The relationship type (e.g., 'STATE_CHANGE').
        object (str): The object node or value (e.g., 'Acidic_Stress').
    """

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., description="The subject node ID, e.g., 'Bioreactor-01'")
    predicate: str = Field(..., description="The relationship type, e.g., 'STATE_CHANGE'")
    object: str = Field(..., description="The object node or value, e.g., 'Acidic_Stress'")


class TwinUpdate(BaseModel):
    """Payload for updating the Digital Twin (Graph Nexus).

    Attributes:
        entity_id (str): ID of the entity being updated.
        timestamp (str): ISO 8601 Timestamp of the update.
        properties (Dict[str, Any]): Dictionary of property updates.
        derived_facts (List[SemanticFact]): List of facts derived from the update.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(..., description="ID of the entity being updated")
    timestamp: str = Field(..., description="ISO 8601 Timestamp of the update")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Raw property updates, e.g., {'ph': 6.2}")
    derived_facts: List[SemanticFact] = Field(
        default_factory=list, description="List of semantic facts derived from the update"
    )
