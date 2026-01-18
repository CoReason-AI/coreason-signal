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
from typing import Any, Callable, Dict, List, Optional, Protocol

from coreason_signal.schemas import SemanticFact, TwinUpdate
from coreason_signal.utils.logger import logger


class GraphConnector(Protocol):
    """
    Protocol for the external Graph Nexus connector.
    """

    def update_node(self, update: TwinUpdate) -> None:
        """
        Send a TwinUpdate to the Graph Nexus.
        """
        ...


class TwinSyncer:
    """
    Synchronizes local state with the Digital Twin (Graph Nexus).
    Implements Delta Throttling and Fact Promotion.
    """

    def __init__(
        self,
        connector: GraphConnector,
        default_sigma_threshold: float = 0.05,
    ) -> None:
        """
        Initialize the Twin Syncer.

        Args:
            connector: Interface to the Graph Nexus.
            default_sigma_threshold: Default relative change required to trigger an update (0.05 = 5%).
        """
        self.connector = connector
        self.default_sigma_threshold = default_sigma_threshold
        # Cache structure: {entity_id: {property_name: last_value}}
        self._state_cache: Dict[str, Dict[str, float]] = {}
        # Fact rules: {property_name: [rule_function]}
        self._fact_rules: Dict[str, List[Callable[[str, Any], Optional[SemanticFact]]]] = {}

    def register_fact_rule(self, property_name: str, rule: Callable[[str, Any], Optional[SemanticFact]]) -> None:
        """
        Register a rule to derive semantic facts from a property.

        Args:
            property_name: The property to listen to (e.g., "ph").
            rule: A function taking (entity_id, value) and returning a SemanticFact or None.
        """
        if property_name not in self._fact_rules:
            self._fact_rules[property_name] = []
        self._fact_rules[property_name].append(rule)

    def _should_sync(self, entity_id: str, property_name: str, value: float, threshold: float) -> bool:
        """
        Check if the value change is significant enough to sync.
        Uses Delta Throttling.
        """
        # Always sync special values (NaN, Inf) or if it's the first value
        if math.isnan(value) or math.isinf(value):
            return True

        if entity_id not in self._state_cache:
            self._state_cache[entity_id] = {}
            return True  # Always sync first value

        if property_name not in self._state_cache[entity_id]:
            return True  # Always sync first value for this property

        last_value = self._state_cache[entity_id][property_name]

        # If last value was NaN/Inf and new is valid, sync.
        if math.isnan(last_value) or math.isinf(last_value):
            return True

        # Avoid division by zero
        if last_value == 0:
            return abs(value - last_value) > 1e-6

        # Calculate relative change
        delta = abs(value - last_value) / abs(last_value)
        return delta >= threshold

    def sync_state(
        self,
        entity_id: str,
        property_name: str,
        value: float,
        timestamp: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Attempt to sync a state change to the Digital Twin.

        Args:
            entity_id: The ID of the entity (e.g., "Bioreactor-01").
            property_name: The property being updated (e.g., "ph").
            value: The new value.
            timestamp: ISO 8601 timestamp.
            threshold: Optional override for sigma threshold.

        Returns:
            True if sync occurred, False if throttled.
        """
        eff_threshold = threshold if threshold is not None else self.default_sigma_threshold

        if not self._should_sync(entity_id, property_name, value, eff_threshold):
            logger.debug(f"Throttled update for {entity_id}.{property_name}: {value}")
            return False

        # Fact Promotion
        facts = self._derive_facts(entity_id, property_name, value)

        # Create Update Payload
        update = TwinUpdate(
            entity_id=entity_id,
            timestamp=timestamp,
            properties={property_name: value},
            derived_facts=facts,
        )

        # Sync
        try:
            self.connector.update_node(update)
            logger.info(f"Synced {entity_id}.{property_name} = {value} ({len(facts)} facts)")

            # CRITICAL: Only update cache AFTER successful sync
            # This ensures that if the network fails, we will retry (not throttle) the next time.
            if entity_id not in self._state_cache:
                self._state_cache[entity_id] = {}
            self._state_cache[entity_id][property_name] = value

            return True
        except Exception as e:
            logger.error(f"Failed to sync twin update for {entity_id}: {e}")
            return False

    def _derive_facts(self, entity_id: str, property_name: str, value: float) -> List[SemanticFact]:
        """
        Apply registered rules to derive facts.
        """
        facts: List[SemanticFact] = []
        if property_name in self._fact_rules:
            for rule in self._fact_rules[property_name]:
                try:
                    fact = rule(entity_id, value)
                    if fact:
                        facts.append(fact)
                except Exception as e:
                    logger.warning(f"Fact rule failed for {property_name}: {e}")
        return facts
