# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Optional

from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import AgentReflex, LogEvent
from coreason_signal.utils.logger import logger


class ReflexEngine:
    """
    The Edge Agent's "Reflex Arc".
    Handles RAG-based decision making by querying local SOPs.
    """

    def __init__(self, persistence_path: str = "./data/lancedb", vector_store: Optional[LocalVectorStore] = None):
        """
        Initialize the Reflex Engine.

        Args:
            persistence_path: Path to the LanceDB directory (used if vector_store is not provided).
            vector_store: Optional injected LocalVectorStore instance.
        """
        if vector_store:
            self._vector_store = vector_store
        else:
            self._vector_store = LocalVectorStore(db_path=persistence_path)

    def decide(self, event: LogEvent) -> Optional[AgentReflex]:
        """
        Query the SOPs based on the log event and return a reflex action.

        Args:
            event: The structured log event.

        Returns:
            The AgentReflex from the most relevant SOP, or None.
        """
        # 1. Check if the event is an error.
        #    Note: Real systems might have more complex logic.
        #    The current tests imply we should look for "ERROR".
        #    However, one test 'test_decide_ignores_non_error' explicitly checks this.
        if event.level != "ERROR":
            return None

        # 2. Extract context from the event
        context = event.message
        if not context or not context.strip():
            return None

        # 3. Query the vector store
        try:
            sops = self._vector_store.query(context, k=1)
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            return None

        if not sops:
            logger.info(f"No relevant SOP found for context: '{context}'")
            return None

        best_sop = sops[0]
        logger.info(f"Matched SOP: {best_sop.id} ({best_sop.title})")

        # 4. Return the reflex
        if best_sop.associated_reflex:
            # We might want to inject specific reflex ID or reason if missing
            # But for now return what's in the SOP.
            return best_sop.associated_reflex

        # If SOP has no specific reflex but was matched, maybe we default to NOTIFY?
        # The test 'test_decide_default_action' expects 'NOTIFY' if no associated reflex.
        return AgentReflex(
            action_name="NOTIFY",
            parameters={"event_id": event.id, "sop_id": best_sop.id},
            reasoning=f"Matched SOP {best_sop.id} but no specific reflex defined.",
        )
