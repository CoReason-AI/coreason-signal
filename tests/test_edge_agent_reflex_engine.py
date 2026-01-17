import datetime
import time
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


def test_reflex_engine_init_default() -> None:
    """Test initializing ReflexEngine without injecting vector store."""
    with patch("coreason_signal.edge_agent.reflex_engine.LocalVectorStore") as mock_lvs_cls:
        engine = ReflexEngine(persistence_path="/tmp/test_db")
        mock_lvs_cls.assert_called_once_with(db_path="/tmp/test_db")
        assert engine._vector_store == mock_lvs_cls.return_value


def test_decide_ignores_non_error(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    event = LogEvent(
        id="evt-001",
        timestamp=datetime.datetime.now().isoformat(),
        level="INFO",
        source="test",
        message="Everything is fine",
    )
    reflex = engine.decide(event)
    assert reflex is None
    mock_vector_store.query.assert_not_called()


def test_decide_no_sop_found(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.return_value = []

    event = LogEvent(
        id="evt-error-1",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="Unknown error",
    )
    reflex = engine.decide(event)
    assert reflex is None
    mock_vector_store.query.assert_called_once_with("Unknown error", k=1)


def test_decide_sop_found(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    reflex_action = AgentReflex(action_name="RETRY", parameters={"speed": 0.5}, reasoning="SOP-104 matches error.")
    sop = SOPDocument(
        id="SOP-104",
        title="Vacuum Error",
        content="Retry at 50% speed.",
        metadata={"suggested_action": "RETRY"},
        associated_reflex=reflex_action,
    )
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(
        id="evt-vac-1",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="LiquidHandler",
        message="ERR_VACUUM_LOW",
    )

    reflex = engine.decide(event)

    assert reflex is not None
    assert isinstance(reflex, AgentReflex)
    assert reflex.action_name == "RETRY"
    # Note: reasoning might be from the SOP reflex or constructed.
    # Here we expect it to be the one from the SOP.
    assert reflex.reasoning == "SOP-104 matches error."


def test_decide_default_action(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    # SOP without associated_reflex
    sop = SOPDocument(id="SOP-Generic", title="Generic Error", content="Call supervisor.", metadata={})
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(
        id="evt-gen-1", timestamp=datetime.datetime.now().isoformat(), level="ERROR", source="test", message="Error"
    )

    reflex = engine.decide(event)
    assert reflex is not None
    assert reflex.action_name == "NOTIFY"
    assert reflex.parameters["sop_id"] == "SOP-Generic"


def test_decide_vector_store_exception(mock_vector_store: MagicMock) -> None:
    """Test robustness against vector store failures."""
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.side_effect = RuntimeError("DB Connection Failed")

    event = LogEvent(
        id="evt-crit-1",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="Critical Failure",
    )

    # Should not raise exception, but log it and return None
    reflex = engine.decide(event)
    assert reflex is None


def test_decide_empty_message(mock_vector_store: MagicMock) -> None:
    """Test handling of empty log messages."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    event = LogEvent(
        id="evt-empty",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="   ",  # Whitespace only
    )

    reflex = engine.decide(event)
    assert reflex is None
    mock_vector_store.query.assert_not_called()


def test_decide_multiple_sops_prioritization(mock_vector_store: MagicMock) -> None:
    """Test that the engine picks the first SOP if multiple are returned."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    reflex_a = AgentReflex(action_name="A", reasoning="A")
    reflex_b = AgentReflex(action_name="B", reasoning="B")

    sop1 = SOPDocument(id="SOP-A", title="A", content="A", associated_reflex=reflex_a)
    sop2 = SOPDocument(id="SOP-B", title="B", content="B", associated_reflex=reflex_b)

    # Mock returning sorted by relevance (best first)
    mock_vector_store.query.return_value = [sop1, sop2]

    event = LogEvent(
        id="evt-ambig",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="Ambiguous error",
    )

    reflex = engine.decide(event)
    assert reflex is not None
    assert reflex.action_name == "A"


def test_decide_watchdog_timeout(mock_vector_store: MagicMock) -> None:
    """Test that decide returns PAUSE if logic takes > 0.2s."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    # Mock _decide_logic to sleep longer than the timeout
    # We patch the instance method
    with patch.object(engine, "_decide_logic", side_effect=lambda e: time.sleep(0.3)):
        event = LogEvent(
            id="evt-timeout",
            timestamp=datetime.datetime.now().isoformat(),
            level="ERROR",
            source="test",
            message="Slow query",
        )

        start_time = time.time()
        reflex = engine.decide(event)
        duration = time.time() - start_time

        # Ensure we returned reasonably quickly (checking upper bound)
        # It should be close to 0.2s, definitely less than the 0.3s sleep
        assert duration < 0.25

        assert reflex is not None
        assert reflex.action_name == "PAUSE"
        assert reflex.reasoning == "Watchdog Timeout > 200ms"
        assert reflex.parameters["event_id"] == "evt-timeout"


def test_decide_watchdog_internal_error(mock_vector_store: MagicMock) -> None:
    """Test that decide catches internal thread errors."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    # Mock _decide_logic to raise an unexpected exception
    with patch.object(engine, "_decide_logic", side_effect=RuntimeError("Thread Crash")):
        event = LogEvent(
            id="evt-crash",
            timestamp=datetime.datetime.now().isoformat(),
            level="ERROR",
            source="test",
            message="Crash me",
        )

        reflex = engine.decide(event)
        assert reflex is None
