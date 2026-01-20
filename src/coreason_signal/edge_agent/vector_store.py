# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import List

import lancedb
from fastembed import TextEmbedding

from coreason_signal.schemas import SOPDocument
from coreason_signal.utils.logger import logger


class LocalVectorStore:
    """In-process vector store using LanceDB for Edge Agent RAG.

    Manages the storage and retrieval of Standard Operating Procedures (SOPs)
    using semantic embeddings.

    Attributes:
        db_path (str): URI of the LanceDB database.
        embedding_model_name (str): Name of the embedding model used.
    """

    def __init__(self, db_path: str = "memory://", embedding_model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Initialize the local vector store.

        Args:
            db_path (str): Path to the LanceDB database. Defaults to "memory://" for in-memory.
            embedding_model_name (str): Name of the embedding model to use with FastEmbed.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self._db = lancedb.connect(db_path)
        self._table_name = "sops"
        self._embedding_model = TextEmbedding(model_name=self.embedding_model_name)

    def add_sops(self, sops: List[SOPDocument]) -> None:
        """Embed and store SOP documents in the vector store.

        Args:
            sops (List[SOPDocument]): List of SOPDocument objects to add.
        """
        if not sops:
            return

        logger.info(f"Adding {len(sops)} SOPs to vector store")

        documents = [sop.content for sop in sops]
        # Generate embeddings
        embeddings = list(self._embedding_model.embed(documents))

        data = []
        for i, sop in enumerate(sops):
            item = sop.model_dump()
            item["vector"] = embeddings[i]
            data.append(item)

        # For lancedb >= 0.26.0, list_tables() returns an iterator of table names
        # or a Pydantic model (ListTablesResponse) depending on the connection type.
        # We handle this by checking if it has a .tables attribute or iterating.
        # Note: table_names() is deprecated but returns a list of strings directly.
        # To be safe and forward compatible:
        existing_tables = []
        try:
            # Try new API first
            tables_response = self._db.list_tables()
            if hasattr(tables_response, "tables"):
                existing_tables = tables_response.tables
            else:
                existing_tables = list(tables_response)  # pragma: no cover
        except Exception:  # pragma: no cover
            # Fallback to deprecated if needed or just empty
            existing_tables = []  # pragma: no cover

        if self._table_name in existing_tables:
            table = self._db.open_table(self._table_name)
            table.add(data)
        else:
            self._db.create_table(self._table_name, data=data)

    def query(self, query_text: str, k: int = 3) -> List[SOPDocument]:
        """Perform a semantic search for SOPs.

        Args:
            query_text (str): The query string (e.g., error message context).
            k (int): Number of nearest neighbors to return.

        Returns:
            List[SOPDocument]: List of SOPDocument objects matching the query.
        """
        existing_tables = []
        try:
            tables_response = self._db.list_tables()
            if hasattr(tables_response, "tables"):
                existing_tables = tables_response.tables
            else:
                existing_tables = list(tables_response)  # pragma: no cover
        except Exception:  # pragma: no cover
            existing_tables = []  # pragma: no cover

        if self._table_name not in existing_tables:
            logger.warning("Query attempted on empty vector store")
            return []

        table = self._db.open_table(self._table_name)

        # Embed the query
        query_embedding = list(self._embedding_model.embed([query_text]))[0]

        results = table.search(query_embedding).limit(k).to_list()

        sops = []
        for res in results:
            sop_data = {
                "id": res["id"],
                "title": res["title"],
                "content": res["content"],
                "metadata": res["metadata"],
                "associated_reflex": res.get("associated_reflex"),
            }
            sops.append(SOPDocument(**sop_data))

        return sops
