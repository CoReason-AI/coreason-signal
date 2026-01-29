import shutil
import tempfile
from typing import Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import SOPDocument


@pytest.fixture
def mock_embedding_model() -> Generator[MagicMock, None, None]:
    with patch("coreason_signal.edge_agent.vector_store.TextEmbedding") as MockTextEmbedding:
        mock_instance = MockTextEmbedding.return_value

        # Mock embed to return a fake vector
        def side_effect(documents: List[str]) -> Generator[List[float], None, None]:
            for _ in documents:
                yield np.random.rand(384).tolist()

        mock_instance.embed.side_effect = side_effect
        yield mock_instance


@pytest.fixture
def test_db_path() -> Generator[str, None, None]:
    """Create a temp directory for the database."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


def test_vector_store_init(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test initialization of LocalVectorStore."""
    store = LocalVectorStore(db_path=test_db_path)
    assert store.db_path == test_db_path
    assert store._db is not None


def test_add_sops(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test adding SOPs to the store."""
    store = LocalVectorStore(db_path=test_db_path)
    sops = [SOPDocument(id="SOP-001", title="Test SOP", content="This is a test content.", metadata={"type": "test"})]
    store.add_sops(sops)

    # Check if table exists by trying to open it
    try:
        table = store._db.open_table("sops")
        assert len(table) == 1
    except ValueError as e:
        pytest.fail(f"Table 'sops' should exist. Error: {e}")


def test_add_sops_append(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test adding SOPs to the store when it already exists (append)."""
    store = LocalVectorStore(db_path=test_db_path)
    sops1 = [
        SOPDocument(id="1", title="A", content="Apple", metadata={}),
    ]
    store.add_sops(sops1)

    sops2 = [
        SOPDocument(id="2", title="B", content="Banana", metadata={}),
    ]
    store.add_sops(sops2)

    table = store._db.open_table("sops")
    assert len(table) == 2


def test_query_sops(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test querying SOPs."""
    store = LocalVectorStore(db_path=test_db_path)

    # Add dummy data
    sops = [
        SOPDocument(id="1", title="A", content="Apple", metadata={}),
        SOPDocument(id="2", title="B", content="Banana", metadata={}),
    ]
    store.add_sops(sops)

    results = store.query("fruit", k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], SOPDocument)


def test_query_empty_store(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test querying an empty store."""
    store = LocalVectorStore(db_path=test_db_path)
    results = store.query("test")
    assert results == []


def test_add_empty_list(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test adding an empty list of SOPs."""
    store = LocalVectorStore(db_path=test_db_path)
    store.add_sops([])
    # Table should not be created if it didn't exist
    try:
        store._db.open_table("sops")
        exists = True
    except ValueError:
        exists = False
    assert not exists


def test_persistence(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test that data persists across store instances sharing the same path."""
    store1 = LocalVectorStore(db_path=test_db_path)
    sops = [SOPDocument(id="P1", title="Persistent", content="Data", metadata={})]
    store1.add_sops(sops)

    # Re-open in a new instance
    store2 = LocalVectorStore(db_path=test_db_path)
    results = store2.query("Data", k=1)
    assert len(results) == 1
    assert results[0].id == "P1"


def test_query_limit_exceeds_count(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test querying with k larger than the number of documents."""
    store = LocalVectorStore(db_path=test_db_path)
    sops = [SOPDocument(id="1", title="One", content="One", metadata={})]
    store.add_sops(sops)

    results = store.query("One", k=10)
    assert len(results) == 1
    assert results[0].id == "1"


def test_add_sops_special_characters(test_db_path: str, mock_embedding_model: MagicMock) -> None:
    """Test adding SOPs with special characters."""
    store = LocalVectorStore(db_path=test_db_path)
    content_with_special = "Special chars: ñ, ü, é, 🚀, €, 100%"
    sops = [SOPDocument(id="SPEC-1", title="Special", content=content_with_special, metadata={})]
    store.add_sops(sops)

    results = store.query("Special", k=1)
    assert len(results) == 1
    assert results[0].content == content_with_special
