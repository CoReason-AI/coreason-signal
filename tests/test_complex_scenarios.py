import datetime
from unittest.mock import MagicMock

import pytest

from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import LogEvent, SOPDocument


@pytest.fixture  # type: ignore[misc]
def mock_vector_store() -> MagicMock:
    return MagicMock()


def test_unicode_log_event(mock_vector_store: MagicMock) -> None:
    """
    Test handling of log events containing Unicode characters and emojis.
    Verifies that the text is passed correctly to the vector store query.
    """
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.return_value = []

    unicode_message = "🔥 Engine Overheat Error: 溫度过高 (Temperature too high)"
    event = LogEvent(
        id="evt-unicode",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="Bioreactor-β",
        message=unicode_message,
    )

    engine.decide(event)

    # Ensure the exact unicode string was passed to the query
    mock_vector_store.query.assert_called_once_with(unicode_message, k=1)


def test_long_log_message(mock_vector_store: MagicMock) -> None:
    """
    Test handling of extremely long log messages.
    Ensures that the system does not crash and attempts to query.
    """
    engine = ReflexEngine(vector_store=mock_vector_store)
    mock_vector_store.query.return_value = []

    # Create a 10KB message
    long_message = "ERROR_CODE_X " * 1000
    event = LogEvent(
        id="evt-long",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="Sequencer",
        message=long_message,
    )

    engine.decide(event)

    mock_vector_store.query.assert_called_once_with(long_message, k=1)


def test_sop_recovery_partial_data(mock_vector_store: MagicMock) -> None:
    """
    Test scenario where the Vector Store returns a valid SOP match,
    but that SOP triggers an internal logic edge case (like missing reflex).
    This simulates a 'partial' knowledge hit.
    """
    engine = ReflexEngine(vector_store=mock_vector_store)

    # SOP with no associated reflex (valid schema, but minimal logic)
    sop = SOPDocument(
        id="SOP-Minimal",
        title="Minimal Info",
        content="Something happened.",
        # No associated_reflex
    )
    mock_vector_store.query.return_value = [sop]

    event = LogEvent(
        id="evt-partial",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="Centrifuge",
        message="Vibration detected",
    )

    reflex = engine.decide(event)

    assert reflex is not None
    assert reflex.action_name == "NOTIFY"
    assert "no specific reflex defined" in reflex.reasoning
    assert reflex.parameters["sop_id"] == "SOP-Minimal"


def test_vector_store_returns_invalid_type_graceful_fail(mock_vector_store: MagicMock) -> None:
    """
    Test scenario where the vector store query returns an object that is NOT an SOPDocument
    (e.g., if dynamic typing goes wrong or mock is bad).
    The ReflexEngine expects List[SOPDocument].
    If it gets something else, it might crash accessing attributes.
    """
    engine = ReflexEngine(vector_store=mock_vector_store)

    # Mock returns a list of plain dicts instead of SOPDocument objects
    # This simulates a potential issue if the vector store layer failed to validate/convert.
    mock_vector_store.query.return_value = [{"id": "bad", "title": "bad"}]  # Not an object

    event = LogEvent(
        id="evt-type-fail",
        timestamp=datetime.datetime.now().isoformat(),
        level="ERROR",
        source="Mixer",
        message="Speed error",
    )

    # We expect this to raise an AttributeError inside decide() when accessing .associated_reflex
    # OR we want to verify if we SHOULD wrap this in try/except.
    # The requirement said "wrap vector store queries in try/except".
    # But this error happens AFTER the query returns, during processing.

    with pytest.raises(AttributeError):
        engine.decide(event)

    # Note: If we want the engine to be robust against THIS too, we would need to catch Exception broadly.
    # Given the robust requirements, maybe we should?
    # But strictly, the vector store contract says it returns List[SOPDocument].
    # So this test confirms that violating the contract causes a crash.
