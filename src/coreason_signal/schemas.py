from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DeviceDefinition(BaseModel):
    """
    Hardware abstraction layer mapping for SiLA 2 and legacy instruments.
    """

    id: str  # e.g., "LiquidHandler-01"
    driver_type: str  # e.g., "SiLA2", "SerialWrapper", "VisionWrapper"
    endpoint: HttpUrl  # e.g., "https://192.168.1.50:50052"
    capabilities: List[str]  # e.g., ["Transfer", "Wash", "Heater"]

    # Edge AI Config
    edge_agent_model: str  # e.g., "phi-4-quantized.onnx"
    allowed_reflexes: List[str]  # e.g., ["RETRY", "PAUSE", "ABORT"]


class SoftSensorModel(BaseModel):
    """
    Configuration for physics-informed neural networks (PINNs) acting as virtual sensors.
    """

    id: str  # e.g., "model_titer_pred_v2"
    input_sensors: List[str]  # e.g., ["ph", "do2", "agitation"]
    target_variable: str  # e.g., "titer_g_L"
    physics_constraints: Dict[str, str]  # e.g., {"min_titer": "0.0"}
    model_artifact: bytes  # The ONNX file

    @field_validator("physics_constraints")
    @classmethod
    def validate_constraint_values(cls, v: Dict[str, str]) -> Dict[str, str]:
        """
        Ensures that all values in physics_constraints are parsable as numbers (float).
        """
        for key, value in v.items():
            try:
                float(value)
            except ValueError:
                raise ValueError(f"Constraint value for '{key}' must be a numeric string, got '{value}'") from None
        return v


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
    level: str = Field(..., description="Log level, e.g., 'INFO', 'ERROR'")
    source: str = Field(..., description="Source component/instrument ID")
    message: str = Field(..., description="The semantic log message, e.g., 'Vacuum Pressure Low'")
    raw_code: Optional[str] = Field(None, description="Original error code, e.g., 'ERR_0x4F'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SemanticFact(BaseModel):
    """
    Represents a derived semantic fact (Subject-Predicate-Object) for the Knowledge Graph.
    e.g. (:Bioreactor)-[:STATE_CHANGE]->(Acidic_Stress)
    """

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., description="The subject node ID, e.g., 'Bioreactor-01'")
    predicate: str = Field(..., description="The relationship type, e.g., 'STATE_CHANGE'")
    object: str = Field(..., description="The object node or value, e.g., 'Acidic_Stress'")


class TwinUpdate(BaseModel):
    """
    Payload for updating the Digital Twin (Graph Nexus).
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(..., description="ID of the entity being updated")
    timestamp: str = Field(..., description="ISO 8601 Timestamp of the update")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Raw property updates, e.g., {'ph': 6.2}")
    derived_facts: List[SemanticFact] = Field(
        default_factory=list, description="List of semantic facts derived from the update"
    )
