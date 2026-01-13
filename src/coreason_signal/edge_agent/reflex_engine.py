# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import List, Optional

import lancedb
from fastembed import TextEmbedding
from lancedb.table import Table

from coreason_signal.schemas import AgentReflex, SOPDocument
from coreason_signal.utils.logger import logger


class ReflexEngine:
    """
    The Edge Agent's "Reflex Arc".
    Handles RAG-based decision making by querying local SOPs.
    """

    TABLE_NAME = "sops"

    def __init__(self, persistence_path: str = "./data/lancedb"):
        """
        Initialize the Reflex Engine with a local vector store.

        Args:
            persistence_path: Path to the LanceDB directory.
        """
        self.persistence_path = persistence_path
        self._db = lancedb.connect(self.persistence_path)

        # Initialize embedding model (default BAAI/bge-small-en-v1.5)
        # fastembed downloads the model automatically on first use.
        self._embedding_model = TextEmbedding()

        self._table: Optional[Table] = None
        self._init_table()

    def _get_table_names(self) -> List[str]:
        """Helper to get list of table names robustly."""
        tables_response = self._db.list_tables()
        if isinstance(tables_response, list):
            return tables_response
        elif hasattr(tables_response, "tables"):
            return tables_response.tables  # type: ignore
        else:
            return list(tables_response)

    def _init_table(self) -> None:
        """Initialize or load the SOP table."""
        try:
            tables = self._get_table_names()
            logger.debug(f"Existing tables: {tables}")

            if self.TABLE_NAME in tables:
                self._table = self._db.open_table(self.TABLE_NAME)
                logger.info(f"Loaded existing table: {self.TABLE_NAME}")
            else:
                pass  # Lazy creation
        except Exception as e:
            logger.error(f"Failed to initialize table: {e}")
            raise

    def ingest_sops(self, sops: List[SOPDocument]) -> None:
        """
        Embed and store Standard Operating Procedures.

        Args:
            sops: List of SOPDocument objects to ingest.
        """
        if not sops:
            return

        logger.info(f"Ingesting {len(sops)} SOPs...")

        texts = [sop.content for sop in sops]
        embeddings = list(self._embedding_model.embed(texts))

        data = []
        for sop, vector in zip(sops, embeddings, strict=False):
            item = sop.model_dump()
            item["vector"] = vector.tolist()
            data.append(item)

        # Refresh table list
        tables = self._get_table_names()
        if self.TABLE_NAME in tables:
            self._table = self._db.open_table(self.TABLE_NAME)
            self._table.add(data)
        else:
            self._table = self._db.create_table(self.TABLE_NAME, data=data)

        logger.info("Ingestion complete.")

    def decide(self, context: str) -> Optional[AgentReflex]:
        """
        Query the SOPs based on the context (e.g., error message) and return a reflex action.

        Args:
            context: The semantic error message or context string.

        Returns:
            The AgentReflex from the most relevant SOP, or None if no relevant SOP is found.
        """
        if self._table is None:
            tables = self._get_table_names()
            if self.TABLE_NAME in tables:
                self._table = self._db.open_table(self.TABLE_NAME)
            else:
                logger.warning("SOP table not found. Cannot make a decision.")
                return None

        # Embed the query
        query_embedding = list(self._embedding_model.embed([context]))[0]

        # Search
        # We assume a limit of 1 for the single best reflex
        results = self._table.search(query_embedding).metric("cosine").limit(1).to_pydantic(SOPDocument)

        if not results:
            logger.info(f"No relevant SOP found for context: '{context}'")
            return None

        best_sop = results[0]
        # In a real system, we would check a similarity score threshold here.
        # fastembed + lancedb usually returns L2 distance or cosine distance.
        # For this atomic unit, we assume the top result is good enough if it exists.

        logger.info(f"Matched SOP: {best_sop.id} ({best_sop.title})")

        if best_sop.associated_reflex:
            return best_sop.associated_reflex  # type: ignore[no-any-return]

        logger.info(f"SOP {best_sop.id} has no associated reflex.")
        return None
