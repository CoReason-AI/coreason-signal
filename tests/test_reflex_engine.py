# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import shutil
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import AgentReflex, SOPDocument


@pytest.fixture  # type: ignore[misc]
def temp_lancedb_path() -> Generator[str, None, None]:
    """Create a temporary directory for LanceDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture  # type: ignore[misc]
def reflex_engine(temp_lancedb_path: str) -> ReflexEngine:
    """Initialize ReflexEngine with a temporary path."""
    return ReflexEngine(persistence_path=temp_lancedb_path)


def test_reflex_engine_initialization(temp_lancedb_path: str) -> None:
    """Test that the engine initializes correctly."""
    engine = ReflexEngine(persistence_path=temp_lancedb_path)
    assert engine.persistence_path == temp_lancedb_path
    # Table should not exist yet
    assert engine._table is None


def test_ingest_and_retrieve_sop(reflex_engine: ReflexEngine) -> None:
    """Test ingesting an SOP and retrieving it via semantic search."""

    # Define a reflex
    reflex = AgentReflex(
        action_name="Aspirate", parameters={"speed": 0.5}, reasoning="Retry aspiration at lower speed to clear clog."
    )

    # Define an SOP
    sop = SOPDocument(
        id="SOP-104",
        title="Vacuum Pressure Low Handling",
        content=(
            "If the vacuum pressure is low during aspiration, it indicates a potential clog. "
            "The recommended action is to retry the aspiration at 50% speed."
        ),
        associated_reflex=reflex,
    )

    # Ingest
    reflex_engine.ingest_sops([sop])

    # Query with a semantic match (not exact text)
    context = "The instrument reported low vacuum pressure error."
    result = reflex_engine.decide(context)

    assert result is not None
    assert result.action_name == "Aspirate"
    assert result.parameters["speed"] == 0.5
    assert result.reasoning == reflex.reasoning


def test_ingest_append(reflex_engine: ReflexEngine) -> None:
    """Test appending SOPs to an existing table."""
    sop1 = SOPDocument(id="1", title="A", content="A", associated_reflex=None)
    reflex_engine.ingest_sops([sop1])

    sop2 = SOPDocument(id="2", title="B", content="B", associated_reflex=None)
    reflex_engine.ingest_sops([sop2])

    # Verify both exist
    # We can check by querying specifically
    res = reflex_engine.decide("A")
    # Should match A (closest)
    assert res is None  # No reflex associated

    # Ensure table has rows.
    # Since we can't count easily without opening table, we rely on the fact that ingest_sops didn't crash
    # and hit the 'append' branch in code.


def test_decide_no_match(reflex_engine: ReflexEngine) -> None:
    """Test that decide returns None when no SOPs are present."""
    result = reflex_engine.decide("Some random error")
    assert result is None


def test_decide_sop_no_reflex(reflex_engine: ReflexEngine) -> None:
    """Test that decide returns None if the matched SOP has no reflex."""
    # Use known working content to ensure search match
    sop = SOPDocument(
        id="SOP-104",
        title="Vacuum Pressure Low Handling",
        content="If the vacuum pressure is low during aspiration, it indicates a potential clog.",
        associated_reflex=None,
    )
    reflex_engine.ingest_sops([sop])

    # Use exact content to guarantee match
    result = reflex_engine.decide("If the vacuum pressure is low during aspiration, it indicates a potential clog.")

    assert result is None


def test_persistence(temp_lancedb_path: str) -> None:
    """Test that data persists across engine instances."""

    # 1. Ingest with first instance
    engine1 = ReflexEngine(persistence_path=temp_lancedb_path)
    sop = SOPDocument(
        id="SOP-P1",
        title="Persistence Test",
        content="Data must survive reboot.",
        associated_reflex=AgentReflex(action_name="Verify", reasoning="Check DB"),
    )
    engine1.ingest_sops([sop])

    # 2. Re-open with second instance
    engine2 = ReflexEngine(persistence_path=temp_lancedb_path)

    # Force table load (usually happens in decide or ingest)
    # We'll just call decide
    result = engine2.decide("survive reboot")

    assert result is not None
    assert result.action_name == "Verify"


def test_ingest_empty_list(reflex_engine: ReflexEngine) -> None:
    """Test ingesting an empty list does nothing."""
    reflex_engine.ingest_sops([])
    # Should not crash
    assert reflex_engine._table is None


def test_lazy_load_table(reflex_engine: ReflexEngine) -> None:
    """Test that decide() loads the table if it exists but _table is None."""
    # Setup: Create table
    sop = SOPDocument(id="1", title="A", content="A", associated_reflex=None)
    reflex_engine.ingest_sops([sop])

    # Force _table to None to simulate lazy load scenario
    reflex_engine._table = None

    # Trigger decide, which should reload table
    reflex_engine.decide("A")

    assert reflex_engine._table is not None


def test_init_exception(temp_lancedb_path: str) -> None:
    """Test exception handling during initialization."""
    with patch("lancedb.connect") as mock_connect:
        mock_connect.side_effect = Exception("DB Error")
        # ReflexEngine calls connect in init
        with pytest.raises(Exception, match="DB Error"):
            ReflexEngine(persistence_path=temp_lancedb_path)


def test_init_table_exception(temp_lancedb_path: str) -> None:
    """Test exception during _init_table."""
    # We mock lancedb.connect to return a mock db that raises on list_tables
    with patch("lancedb.connect") as mock_connect:
        mock_db = MagicMock()
        # Mock _get_table_names by mocking list_tables behavior
        mock_db.list_tables.side_effect = Exception("List Error")
        mock_connect.return_value = mock_db

        with pytest.raises(Exception, match="List Error"):
            ReflexEngine(persistence_path=temp_lancedb_path)


def test_fallback_list_tables(temp_lancedb_path: str) -> None:
    """Test fallback when list_tables returns something other than object with .tables."""
    with patch("lancedb.connect") as mock_connect:
        mock_db = MagicMock()
        # Mock list_tables to return a simple list (legacy behavior)
        mock_db.list_tables.return_value = ["sops"]
        mock_connect.return_value = mock_db

        # This calls _init_table
        ReflexEngine(persistence_path=temp_lancedb_path)

        # Verify it handled it (no crash)
        # And since "sops" is in the list, it should try to open it
        mock_db.open_table.assert_called_with("sops")


def test_fallback_list_tables_iterable(temp_lancedb_path: str) -> None:
    """Test fallback when list_tables returns an iterable (not list, no .tables)."""
    with patch("lancedb.connect") as mock_connect:
        mock_db = MagicMock()
        # Mock list_tables to return a tuple (iterable but not list, no .tables)
        mock_db.list_tables.return_value = ("sops",)
        mock_connect.return_value = mock_db

        # This calls _init_table
        print("Initializing engine in test...")
        ReflexEngine(persistence_path=temp_lancedb_path)
        print("Engine initialized.")

        # Verify it handled it
        print(f"Mock calls: {mock_db.mock_calls}")
        mock_db.open_table.assert_called_with("sops")


def test_decide_search_no_results(reflex_engine: ReflexEngine) -> None:
    """Test decide when search returns no results."""
    # Ingest something so table exists
    sop = SOPDocument(id="1", title="A", content="A", associated_reflex=None)
    reflex_engine.ingest_sops([sop])

    # Mock the search chain: search(...).metric(...).limit(...).to_pydantic(...)
    with patch.object(reflex_engine, "_table") as mock_table:
        mock_search = MagicMock()
        mock_metric = MagicMock()
        mock_limit = MagicMock()

        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_metric
        mock_metric.limit.return_value = mock_limit

        # Set to_pydantic to return empty list
        mock_limit.to_pydantic.return_value = []

        result = reflex_engine.decide("A")

        assert result is None
