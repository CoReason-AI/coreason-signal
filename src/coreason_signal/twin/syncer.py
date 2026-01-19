# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import math
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Protocol

from coreason_signal.schemas import SemanticFact, TwinUpdate
from coreason_signal.utils.logger import logger


class GraphConnector(Protocol):
    """Protocol for the external Graph Nexus connector."""

    def update_node(self, update: TwinUpdate) -> None:
        """Send a TwinUpdate to the Graph Nexus.

        Args:
            update (TwinUpdate): The update payload.
        """
        ...


class TwinSyncer:
    """Synchronizes local state with the Digital Twin (Graph Nexus).

    Implements Delta Throttling to reduce network traffic and Fact Promotion
    to derive semantic insights from raw data.

    Attributes:
        connector (GraphConnector): Interface to the Graph Nexus.
        default_sigma_threshold (float): Default relative change required to trigger an update.
    """

    def __init__(
        self,
        connector: GraphConnector,
        default_sigma_threshold: float = 0.05,
    ) -> None:
        """Initialize the Twin Syncer.

        Args:
            connector (GraphConnector): Interface to the Graph Nexus.
            default_sigma_threshold (float): Default relative change required to trigger an update (0.05 = 5%).
        """
        self.connector = connector
        self.default_sigma_threshold = default_sigma_threshold
        self._lock = threading.RLock()

        # Cache structure: {entity_id: {property_name: last_synced_value}}
        self._last_synced_state: Dict[str, Dict[str, float]] = {}

        # Fact rules: {property_name: [rule_function]}
        self._fact_rules: Dict[str, List[Callable[[str, Any], Optional[SemanticFact]]]] = defaultdict(list)

    def register_fact_rule(self, property_name: str, rule: Callable[[str, Any], Optional[SemanticFact]]) -> None:
        """Register a rule to derive semantic facts from a property.

        Args:
            property_name (str): The property to listen to (e.g., "ph").
            rule (Callable[[str, Any], Optional[SemanticFact]]): A function taking (entity_id, value)
                and returning a SemanticFact or None.
        """
        with self._lock:
            self._fact_rules[property_name].append(rule)

    @staticmethod
    def _is_significant_change(
        old_value: float, new_value: float, threshold: float, zero_tolerance: float = 1e-6
    ) -> bool:
        """Determine if the change between old and new values is significant.

        Handles NaN, Inf, and Zero divisions robustly.

        Args:
            old_value (float): The previous value.
            new_value (float): The new value.
            threshold (float): The percentage threshold for change.
            zero_tolerance (float): Small epsilon to handle near-zero values.

        Returns:
            bool: True if the change is significant, False otherwise.
        """
        # 1. NaN/Inf handling
        if math.isnan(new_value) or math.isinf(new_value):
            return True  # Always sync anomaly
        if math.isnan(old_value) or math.isinf(old_value):
            return True  # Always sync recovery from anomaly

        # 2. Zero handling
        if abs(old_value) < zero_tolerance:
            # If old was ~0, any change > tolerance is significant
            return abs(new_value - old_value) > zero_tolerance

        # 3. Relative change
        delta = abs(new_value - old_value) / abs(old_value)
        return delta >= threshold

    def _should_sync(self, entity_id: str, property_name: str, value: float, threshold: float) -> bool:
        """Check if the value change is significant enough to sync.

        Uses Delta Throttling based on the last synced state.

        Args:
            entity_id (str): The entity ID.
            property_name (str): The property name.
            value (float): The current value.
            threshold (float): The significance threshold.

        Returns:
            bool: True if sync is required.
        """
        # Always sync special values (NaN, Inf) if we don't know the state
        if math.isnan(value) or math.isinf(value):
            return True

        if entity_id not in self._last_synced_state:
            return True  # Always sync first value

        if property_name not in self._last_synced_state[entity_id]:
            return True  # Always sync first value for this property

        last_value = self._last_synced_state[entity_id][property_name]

        return self._is_significant_change(last_value, value, threshold)

    def sync_state(
        self,
        entity_id: str,
        property_name: str,
        value: float,
        timestamp: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """Attempt to sync a state change to the Digital Twin.

        Checks if the change is significant (throttling), derives semantic facts using
        registered rules, and pushes the update via the connector.

        Args:
            entity_id (str): The ID of the entity (e.g., "Bioreactor-01").
            property_name (str): The property being updated (e.g., "ph").
            value (float): The new value.
            timestamp (str): ISO 8601 timestamp.
            threshold (Optional[float]): Optional override for sigma threshold.

        Returns:
            bool: True if sync occurred, False if throttled or failed.
        """
        eff_threshold = threshold if threshold is not None else self.default_sigma_threshold

        with self._lock:
            if not self._should_sync(entity_id, property_name, value, eff_threshold):
                logger.debug(f"Throttled update for {entity_id}.{property_name}: {value}")
                return False

        # Fact Promotion (can run outside lock if rules are thread-safe, but accessing rules list needs lock if dynamic)
        # For safety, let's copy the rules list under lock or just run it. The lock is RLock so re-entry is fine.
        # However, calling external code under lock is generally bad (deadlocks).
        # We will fetch rules under lock, run outside.
        with self._lock:
            rules = list(self._fact_rules.get(property_name, []))

        facts = self._derive_facts_from_rules(rules, entity_id, value, property_name)

        # Create Update Payload
        update = TwinUpdate(
            entity_id=entity_id,
            timestamp=timestamp,
            properties={property_name: value},
            derived_facts=facts,
        )

        # Sync (Network Call - Do NOT hold lock)
        try:
            self.connector.update_node(update)
            logger.info(f"Synced {entity_id}.{property_name} = {value} ({len(facts)} facts)")

            # Update cache AFTER successful sync
            with self._lock:
                if entity_id not in self._last_synced_state:
                    self._last_synced_state[entity_id] = {}
                self._last_synced_state[entity_id][property_name] = value

            return True
        except Exception as e:
            logger.error(f"Failed to sync twin update for {entity_id}: {e}")
            return False

    def _derive_facts_from_rules(
        self,
        rules: List[Callable[[str, Any], Optional[SemanticFact]]],
        entity_id: str,
        value: float,
        property_name: str,
    ) -> List[SemanticFact]:
        """Apply rules to derive facts.

        Args:
            rules (List[Callable]): List of rule functions.
            entity_id (str): The entity ID.
            value (float): The current value.
            property_name (str): The property name.

        Returns:
            List[SemanticFact]: A list of derived facts.
        """
        facts: List[SemanticFact] = []
        for rule in rules:
            try:
                fact = rule(entity_id, value)
                if fact:
                    facts.append(fact)
            except Exception as e:
                logger.warning(f"Fact rule failed for {property_name}: {e}")
        return facts
