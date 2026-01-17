# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import concurrent.futures
from typing import Optional

from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import AgentReflex, LogEvent
from coreason_signal.utils.logger import logger


class ReflexEngine:
    """
    The Edge Agent's "Reflex Arc".
    Handles RAG-based decision making by querying local SOPs.
    """

    def __init__(
        self,
        persistence_path: str = "./data/lancedb",
        vector_store: Optional[LocalVectorStore] = None,
        decision_timeout: float = 0.2,
    ):
        """
        Initialize the Reflex Engine.

        Args:
            persistence_path: Path to the LanceDB directory (used if vector_store is not provided).
            vector_store: Optional injected LocalVectorStore instance.
            decision_timeout: Time in seconds before the Dead Man's Switch triggers (default: 0.2s).
        """
        if vector_store:
            self._vector_store = vector_store
        else:
            self._vector_store = LocalVectorStore(db_path=persistence_path)

        self.decision_timeout = decision_timeout
        # Use a persistent executor to avoid overhead and blocking shutdown issues
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _decide_logic(self, event: LogEvent) -> Optional[AgentReflex]:
        """
        Internal logic for decision making.
        """
        # 1. Check if the event is an error.
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
            return best_sop.associated_reflex

        # If SOP has no specific reflex but was matched, default to NOTIFY
        return AgentReflex(
            action_name="NOTIFY",
            parameters={"event_id": event.id, "sop_id": best_sop.id},
            reasoning=f"Matched SOP {best_sop.id} but no specific reflex defined.",
        )

    def decide(self, event: LogEvent) -> Optional[AgentReflex]:
        """
        Query the SOPs based on the log event and return a reflex action.
        Enforces a 200ms timeout (Dead Man's Switch).

        Args:
            event: The structured log event.

        Returns:
            The AgentReflex from the most relevant SOP, None, or a PAUSE reflex on timeout.
        """
        try:
            future = self._executor.submit(self._decide_logic, event)
            try:
                return future.result(timeout=self.decision_timeout)
            except concurrent.futures.TimeoutError:
                ms_timeout = int(self.decision_timeout * 1000)
                logger.critical(f"Reflex Engine Watchdog Triggered: Decision took >{ms_timeout}ms for event {event.id}")
                return AgentReflex(
                    action_name="PAUSE",
                    reasoning=f"Watchdog Timeout > {ms_timeout}ms",
                    parameters={"event_id": event.id},
                )
            except Exception as e:
                logger.exception(f"Reflex Engine crashed: {e}")
                return None
        except Exception as e:
            # Catch submission errors (e.g., executor shutdown)
            logger.exception(f"Reflex Engine submission failed: {e}")
            return None
