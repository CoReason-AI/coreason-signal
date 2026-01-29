from typing import Any, Dict

import pytest
from coreason_signal.schemas import (
    AgentReflex,
    DeviceDefinition,
    LogEvent,
    SemanticFact,
    SoftSensorModel,
    SOPDocument,
    TwinUpdate,
)
from pydantic import ValidationError


def test_soft_sensor_constraints_edge_cases() -> None:
    """
    Test edge cases for SoftSensorModel physics constraints.
    Verifies handling of special numeric values and invalid strings.
    """
    # 1. Infinity should be valid (float("inf") works)
    model = SoftSensorModel(
        id="model-inf",
        input_sensors=["x"],
        target_variable="y",
        physics_constraints={"max_limit": "inf", "min_limit": "-Infinity"},
        model_artifact=b"data",
    )
    assert float(model.physics_constraints["max_limit"]) == float("inf")

    # 2. Empty string should fail
    with pytest.raises(ValidationError) as exc:
        SoftSensorModel(
            id="model-empty",
            input_sensors=["x"],
            target_variable="y",
            physics_constraints={"val": ""},
            model_artifact=b"data",
        )
    assert "Input should be a valid number" in str(exc.value)

    # 3. Malformed number should fail
    with pytest.raises(ValidationError) as exc:
        SoftSensorModel(
            id="model-malformed",
            input_sensors=["x"],
            target_variable="y",
            physics_constraints={"val": "1.2.3.4"},
            model_artifact=b"data",
        )
    assert "Input should be a valid number" in str(exc.value)


def test_sop_document_complex_roundtrip() -> None:
    """
    Verify strict JSON serialization and deserialization for a complex, nested SOPDocument.
    Checks that nested AgentReflex and dictionaries are preserved perfectly.
    """
    complex_params: Dict[str, Any] = {
        "nested_config": {"threshold": 0.95, "modes": [1, 2, 3]},
        "flags": [True, False, None],
    }

    original_sop = SOPDocument(
        id="SOP-Complex-001",
        title="Complex Recovery",
        content="Detailed instructions...",
        metadata={"version": 2, "tags": ["critical", "automated"]},
        associated_reflex=AgentReflex(
            action="COMPLEX_ACTION",
            parameters=complex_params,
            reasoning="Testing deep nesting.",
        ),
    )

    # Serialize
    json_str = original_sop.model_dump_json()

    # Deserialize
    restored_sop = SOPDocument.model_validate_json(json_str)

    # Verify Equality
    assert restored_sop.id == original_sop.id
    assert restored_sop.associated_reflex is not None
    assert restored_sop.associated_reflex.action == "COMPLEX_ACTION"
    assert restored_sop.associated_reflex.parameters["nested_config"]["threshold"] == 0.95
    assert restored_sop.associated_reflex.parameters["flags"] == [True, False, None]


def test_schema_forbid_extra_fields() -> None:
    """
    Verify that core schemas strictly forbid extra fields to prevent data pollution.
    """
    # LogEvent (extra="forbid")
    with pytest.raises(ValidationError) as exc:
        LogEvent(
            id="evt-1",
            timestamp="2025-01-01T00:00:00Z",
            level="INFO",
            source="src",
            message="msg",
            unexpected_field="fail_me",  # type: ignore[call-arg]
        )
    assert "extra_forbidden" in str(exc.value).lower()

    # TwinUpdate (extra="forbid")
    with pytest.raises(ValidationError) as exc:
        TwinUpdate(
            entity_id="twin-1",
            timestamp="2025-01-01",
            unexpected="fail",  # type: ignore[call-arg]
        )
    assert "extra_forbidden" in str(exc.value).lower()


def test_device_definition_ipv6_endpoint() -> None:
    """
    Test that DeviceDefinition accepts IPv6 addresses in endpoints.
    """
    ipv6_url = "http://[2001:db8:85a3::8a2e:370:7334]:8080"
    device = DeviceDefinition(
        id="IPv6-Device",
        driver_type="SiLA2",
        endpoint=ipv6_url,  # Pydantic expects valid URL string
        capabilities=[],
        edge_agent_model="model",
        allowed_reflexes=[],
    )

    # Pydantic 2.x HttpUrl stringifies cleanly
    assert str(device.endpoint).rstrip("/") == ipv6_url
    assert device.endpoint.port == 8080


def test_log_event_large_payload() -> None:
    """
    Test that LogEvent handles very large payloads without validation issues.
    (Note: Practical limits depend on system memory, but schema shouldn't arbitrarily limit).
    """
    large_message = "A" * 100_000  # 100KB string
    large_metadata = {f"key_{i}": i for i in range(1000)}

    event = LogEvent(
        id="evt-large",
        timestamp="now",
        level="DEBUG",
        source="stress_test",
        message=large_message,
        metadata=large_metadata,
    )

    assert len(event.message) == 100_000
    assert event.metadata["key_999"] == 999


def test_semantic_fact_validation() -> None:
    """
    Test SemanticFact creation and validation.
    """
    fact = SemanticFact(subject="NodeA", predicate="CONNECTED_TO", object="NodeB")
    assert fact.subject == "NodeA"

    # Verify extras forbidden
    with pytest.raises(ValidationError):
        SemanticFact(
            subject="A",
            predicate="P",
            object="O",
            extra="bad",  # type: ignore[call-arg]
        )
