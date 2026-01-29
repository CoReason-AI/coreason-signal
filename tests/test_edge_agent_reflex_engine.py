import datetime
import time
from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext

from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import AgentReflex, LogEvent, SOPDocument


@pytest.fixture
def mock_vector_store() -> MagicMock:
    return MagicMock()


def test_reflex_engine_init(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    assert engine._vector_store == mock_vector_store
    assert engine._executor is not None


def test_reflex_engine_init_default() -> None:
    """Test initializing ReflexEngine with explicit store (Refactored to assume caller responsibility)."""
    # The default behavior was removed in favor of strict DI.
    # We verify that passing a manually created store works.
    with patch("coreason_signal.edge_agent.reflex_engine.LocalVectorStore") as mock_lvs_cls:
        # Caller must instantiate store
        store_instance = mock_lvs_cls.return_value
        engine = ReflexEngine(vector_store=store_instance)
        assert engine._vector_store == store_instance


def test_decide_ignores_non_error(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    event = LogEvent(
        id="evt-001",
        timestamp=datetime.datetime.now().isoformat(),
        level="INFO",
        source="test",
        message="Everything is fine",
    )
    reflex = engine.decide(event, user_context)
    assert reflex is None


def test_reflex_trigger_execution(mock_vector_store: MagicMock) -> None:
    """Test manual trigger execution."""
    engine = ReflexEngine(vector_store=mock_vector_store)
    reflex = AgentReflex(action="TEST", parameters={"p": 1}, reasoning="test")

    with patch.object(engine, "_execute_reflex_logic") as mock_exec:
        engine.trigger(reflex)
        engine._executor.shutdown(wait=True)
        mock_exec.assert_called_once_with(reflex)


def test_reflex_execution_logic(mock_vector_store: MagicMock) -> None:
    """Test the internal execution logic (logging)."""
    engine = ReflexEngine(vector_store=mock_vector_store)
    reflex = AgentReflex(action="TEST", parameters={"p": 1}, reasoning="test")
    # This just ensures it doesn't crash
    engine._execute_reflex_logic(reflex)


def test_decide_missing_context(mock_vector_store: MagicMock) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    event = LogEvent(id="1", timestamp="", level="ERROR", source="t", message="m")
    with pytest.raises(ValueError, match="UserContext is required"):
        engine.decide(event, None)  # type: ignore[arg-type]
    mock_vector_store.query.assert_not_called()


def test_decide_no_sop_found(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.return_value = []

    event = LogEvent(
        id="evt-error-1",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="Unknown error",
    )
    reflex = engine.decide(event, user_context)
    assert reflex is None
    mock_vector_store.query.assert_called_once_with("Unknown error", k=1)


def test_decide_sop_found(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    reflex_action = AgentReflex(action="RETRY", parameters={"speed": 0.5}, reasoning="SOP-104 matches error.")
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

    reflex = engine.decide(event, user_context)

    assert reflex is not None
    assert isinstance(reflex, AgentReflex)
    assert reflex.action == "RETRY"
    # Note: reasoning might be from the SOP reflex or constructed.
    # Here we expect it to be the one from the SOP.
    assert reflex.reasoning == "SOP-104 matches error."


def test_decide_default_action(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    engine = ReflexEngine(vector_store=mock_vector_store)

    # SOP without associated_reflex
    sop = SOPDocument(id="SOP-Generic", title="Generic Error", content="Call supervisor.", metadata={})
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(
        id="evt-gen-1", timestamp=datetime.datetime.now().isoformat(), level="ERROR", source="test", message="Error"
    )

    reflex = engine.decide(event, user_context)
    assert reflex is not None
    assert reflex.action == "NOTIFY"
    assert reflex.parameters["sop_id"] == "SOP-Generic"


def test_decide_vector_store_exception(mock_vector_store: MagicMock, user_context: UserContext) -> None:
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
    reflex = engine.decide(event, user_context)
    assert reflex is None


def test_decide_empty_message(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """Test handling of empty log messages."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    event = LogEvent(
        id="evt-empty",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="   ",  # Whitespace only
    )

    reflex = engine.decide(event, user_context)
    assert reflex is None
    mock_vector_store.query.assert_not_called()


def test_decide_multiple_sops_prioritization(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """Test that the engine picks the first SOP if multiple are returned."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    reflex_a = AgentReflex(action="A", reasoning="A")
    reflex_b = AgentReflex(action="B", reasoning="B")

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

    reflex = engine.decide(event, user_context)
    assert reflex is not None
    assert reflex.action == "A"


def test_decide_watchdog_timeout(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """Test that decide returns PAUSE if logic takes > 0.2s."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    # Mock _decide_logic to sleep longer than the timeout (0.5s > 0.2s)
    # This helps differentiate between 'wait for completion' vs 'overhead'
    with patch.object(engine, "_decide_logic", side_effect=lambda e: time.sleep(0.5)):
        event = LogEvent(
            id="evt-timeout",
            timestamp=datetime.datetime.now().isoformat(),
            level="ERROR",
            source="test",
            message="Slow query",
        )

        start_time = time.time()
        reflex = engine.decide(event, user_context)
        duration = time.time() - start_time

        # If strict timeout works, duration should be ~0.2s + overhead.
        # It must be significantly less than 0.5s to prove we didn't wait.
        assert duration < 0.4

        assert reflex is not None
        assert reflex.action == "PAUSE"
        assert reflex.reasoning == "Watchdog Timeout > 200ms"
        assert reflex.parameters["event_id"] == "evt-timeout"


def test_decide_watchdog_internal_error(mock_vector_store: MagicMock, user_context: UserContext) -> None:
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

        reflex = engine.decide(event, user_context)
        assert reflex is None


def test_decide_executor_failure(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """Test that decide handles failures during task submission."""
    engine = ReflexEngine(vector_store=mock_vector_store)

    # Mock the executor submit to raise exception
    engine._executor = MagicMock()
    engine._executor.submit.side_effect = RuntimeError("Executor full")

    event = LogEvent(
        id="evt-exec-fail",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="test",
        message="Fail submission",
    )

    reflex = engine.decide(event, user_context)
    assert reflex is None
