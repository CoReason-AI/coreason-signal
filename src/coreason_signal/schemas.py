from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl, field_validator


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


class SOPDocument(BaseModel):
    """
    Standard Operating Procedure (SOP) document for the Edge Agent's RAG system.
    """

    id: str  # e.g., "SOP-104"
    title: str  # e.g., "Vacuum Pressure Low Recovery"
    content: str  # e.g., "For vacuum errors, retry aspiration once at 50% speed."
    metadata: Dict[str, str] = {}  # e.g., {"error_code": "ERR_VACUUM_PRESSURE_LOW"}


class LogEvent(BaseModel):
    """
    Structured log event from an instrument.
    """

    timestamp: datetime
    source: str
    level: str  # e.g., "INFO", "ERROR"
    raw_message: str
    metadata: Dict[str, Any] = {}


class AgentReflex(BaseModel):
    """
    The decision/action output by the Edge Agent.
    """

    reflex_id: str
    action: str  # e.g., "RETRY", "PAUSE", "ABORT"
    parameters: Dict[str, Any] = {}
    reasoning: str
    sop_id: Optional[str] = None
