from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import AgentReflex, LogEvent, SOPDocument


@pytest.fixture  # type: ignore[misc]
def mock_vector_store() -> MagicMock:
    return MagicMock()


def test_reflex_engine_init(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    assert engine._vector_store == mock_vector_store


def test_decide_ignores_non_error(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    event = LogEvent(timestamp=datetime.now(), source="test", level="INFO", raw_message="Everything is fine")
    reflex = engine.decide(event)
    assert reflex is None
    mock_vector_store.query.assert_not_called()


def test_decide_no_sop_found(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.return_value = []

    event = LogEvent(timestamp=datetime.now(), source="test", level="ERROR", raw_message="Unknown error")
    reflex = engine.decide(event)
    assert reflex is None
    mock_vector_store.query.assert_called_once_with("Unknown error", k=1)


def test_decide_sop_found(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    sop = SOPDocument(
        id="SOP-104", title="Vacuum Error", content="Retry at 50% speed.", metadata={"suggested_action": "RETRY"}
    )
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(timestamp=datetime.now(), source="LiquidHandler", level="ERROR", raw_message="ERR_VACUUM_LOW")

    with patch("coreason_signal.edge_agent.reflex_engine.uuid.uuid4", return_value="test-uuid"):
        reflex = engine.decide(event)

    assert reflex is not None
    assert isinstance(reflex, AgentReflex)
    assert reflex.reflex_id == "test-uuid"
    assert reflex.action == "RETRY"
    assert reflex.sop_id == "SOP-104"
    assert "Matched SOP SOP-104" in reflex.reasoning


def test_decide_default_action(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    # SOP without suggested_action
    sop = SOPDocument(id="SOP-Generic", title="Generic Error", content="Call supervisor.", metadata={})
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(timestamp=datetime.now(), source="test", level="ERROR", raw_message="Error")

    reflex = engine.decide(event)
    assert reflex is not None
    assert reflex.action == "NOTIFY"
