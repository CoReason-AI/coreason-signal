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

from coreason_identity.models import UserContext

from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import AgentReflex, LogEvent
from coreason_signal.utils.logger import logger


class ReflexEngine:
    """The Edge Agent's "Reflex Arc" for autonomous decision making.

    It handles RAG-based decision making by querying local Standard Operating Procedures (SOPs)
    stored in a vector database. It implements a "Dead Man's Switch" to ensure safety
    by enforcing strict timeouts on decisions.

    Attributes:
        decision_timeout (float): Max time allowed for a decision before safety override.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        decision_timeout: float = 0.2,
    ):
        """Initialize the Reflex Engine.

        Args:
            vector_store (LocalVectorStore): The vector store instance for retrieving SOPs.
            decision_timeout (float): Time in seconds before the Dead Man's Switch triggers (default: 0.2s).
        """
        self._vector_store = vector_store
        self.decision_timeout = decision_timeout
        # Use a persistent executor to avoid overhead and blocking shutdown issues
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _decide_logic(self, event: LogEvent) -> Optional[AgentReflex]:
        """Internal logic for decision making.

        Args:
            event (LogEvent): The log event to process.

        Returns:
            Optional[AgentReflex]: The determined reflex action, or None if no action needed.
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
            action="NOTIFY",
            parameters={"event_id": event.id, "sop_id": best_sop.id},
            reasoning=f"Matched SOP {best_sop.id} but no specific reflex defined.",
        )

    def decide(self, event: LogEvent, context: UserContext) -> Optional[AgentReflex]:
        """Query the SOPs based on the log event and return a reflex action.

        Enforces a strict timeout (Dead Man's Switch). If the decision logic takes longer
        than `decision_timeout`, a PAUSE reflex is returned to ensure safety.

        Args:
            event (LogEvent): The structured log event.
            context (UserContext): The identity context.

        Returns:
            Optional[AgentReflex]: The AgentReflex from the most relevant SOP,
            None if no relevant SOP found or not an error,
            or a 'PAUSE' AgentReflex on timeout.
        """
        if context is None:
            raise ValueError("UserContext is required for Reflex Engine.")

        logger.debug("Processing reflex event", user_id=context.user_id.get_secret_value(), event_type=event.level)

        try:
            future = self._executor.submit(self._decide_logic, event)
            try:
                return future.result(timeout=self.decision_timeout)
            except concurrent.futures.TimeoutError:
                ms_timeout = int(self.decision_timeout * 1000)
                logger.critical(f"Reflex Engine Watchdog Triggered: Decision took >{ms_timeout}ms for event {event.id}")
                return AgentReflex(
                    action="PAUSE",
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
