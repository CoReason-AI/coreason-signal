import uuid
from typing import Any, Dict, Optional

from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import AgentReflex, LogEvent
from coreason_signal.utils.logger import logger


class ReflexEngine:
    """
    Core decision engine for the Edge Agent.
    Evaluates log events and determines reflex actions using a local vector store (RAG).
    """

    def __init__(self, vector_store: LocalVectorStore) -> None:
        """
        Initialize the ReflexEngine.

        Args:
            vector_store: Instance of LocalVectorStore for retrieving SOPs.
        """
        self._vector_store = vector_store

    def decide(self, event: LogEvent) -> Optional[AgentReflex]:
        """
        Evaluate a log event and decide on an action.

        Args:
            event: The LogEvent to analyze.

        Returns:
            An AgentReflex object if an action is determined, else None.
        """
        # Only act on errors
        if event.level != "ERROR":
            logger.debug(f"Ignoring non-error event: {event.level}")
            return None

        if not event.raw_message.strip():
            logger.warning("Event has empty message, skipping.")
            return None

        logger.info(f"Analyzing error: {event.raw_message}")

        # Query SOPs
        try:
            # Note: In a real implementation, we might parse the error code specifically.
            # Here we use the raw message for semantic search.
            sops = self._vector_store.query(event.raw_message, k=1)
        except Exception as e:
            logger.exception(f"Vector store query failed: {e}")
            return None

        if not sops:
            logger.warning("No relevant SOP found for error.")
            return None

        best_sop = sops[0]
        logger.info(f"Found SOP: {best_sop.id} ({best_sop.title})")

        # Deterministic Logic:
        # Extract action from metadata if available, otherwise default to NOTIFY
        action = best_sop.metadata.get("suggested_action", "NOTIFY")

        # Parse parameters (simplified: assuming they are stored as a stringified dict or similar in metadata,
        # or just hardcoded for this atomic unit).
        # For this iteration, we will look for 'action_params' in metadata, which might be a JSON string,
        # but to keep it simple and type-safe without extra parsing logic yet, we'll use an empty dict
        # unless 'action_params' is explicitly provided as a dict in the mock data.
        # Since metadata is Dict[str, str], we can't store a Dict directly.
        # We will assume no parameters for now unless we implement a parser.
        parameters: Dict[str, Any] = {}

        return AgentReflex(
            reflex_id=str(uuid.uuid4()),
            action=action,
            parameters=parameters,
            reasoning=f"Matched SOP {best_sop.id}: {best_sop.title}. {best_sop.content}",
            sop_id=best_sop.id,
        )
