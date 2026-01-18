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
from unittest.mock import MagicMock

import pytest

from coreason_signal.schemas import SemanticFact, TwinUpdate
from coreason_signal.twin.syncer import TwinSyncer


class MockConnector:
    def update_node(self, update: TwinUpdate) -> None:
        pass


@pytest.fixture  # type: ignore[misc]
def mock_connector() -> MagicMock:
    return MagicMock(spec=MockConnector)


@pytest.fixture  # type: ignore[misc]
def twin_syncer(mock_connector: MagicMock) -> TwinSyncer:
    return TwinSyncer(connector=mock_connector, default_sigma_threshold=0.1)  # 10% threshold


def test_initialization(twin_syncer: TwinSyncer) -> None:
    assert twin_syncer.default_sigma_threshold == 0.1
    assert twin_syncer._state_cache == {}


def test_first_sync_always_passes(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    entity_id = "Bio-1"
    ts = "2023-01-01T12:00:00Z"

    synced = twin_syncer.sync_state(entity_id, "ph", 7.0, ts)

    assert synced is True
    assert twin_syncer._state_cache[entity_id]["ph"] == 7.0
    mock_connector.update_node.assert_called_once()

    call_arg = mock_connector.update_node.call_args[0][0]
    assert isinstance(call_arg, TwinUpdate)
    assert call_arg.entity_id == entity_id
    assert call_arg.properties == {"ph": 7.0}


def test_throttling_logic(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    entity_id = "Bio-1"
    ts = "2023-01-01T12:00:00Z"

    # First sync (baseline)
    twin_syncer.sync_state(entity_id, "ph", 7.0, ts)
    mock_connector.update_node.reset_mock()

    # 7.05 is < 10% change from 7.0 (delta ~0.007) -> Should Throttle
    synced = twin_syncer.sync_state(entity_id, "ph", 7.05, ts)
    assert synced is False
    mock_connector.update_node.assert_not_called()
    assert twin_syncer._state_cache[entity_id]["ph"] == 7.0  # Cache should not update

    # 8.0 is > 10% change from 7.0 -> Should Sync
    synced = twin_syncer.sync_state(entity_id, "ph", 8.0, ts)
    assert synced is True
    mock_connector.update_node.assert_called_once()
    assert twin_syncer._state_cache[entity_id]["ph"] == 8.0  # Cache updates


def test_zero_handling(twin_syncer: TwinSyncer) -> None:
    entity_id = "Bio-1"
    ts = "2023-01-01T12:00:00Z"

    twin_syncer.sync_state(entity_id, "speed", 0.0, ts)

    # Change from 0 to 0.0001
    synced = twin_syncer.sync_state(entity_id, "speed", 0.0001, ts)
    assert synced is True  # Handled by abs check


def test_fact_promotion(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    def ph_rule(entity_id: str, value: float) -> Optional[SemanticFact]:
        if value < 6.5:
            return SemanticFact(subject=entity_id, predicate="HAS_STATE", object="Acidic")
        return None

    twin_syncer.register_fact_rule("ph", ph_rule)

    entity_id = "Bio-1"
    ts = "2023-01-01T12:00:00Z"

    # Sync with rule triggering
    twin_syncer.sync_state(entity_id, "ph", 6.0, ts)

    mock_connector.update_node.assert_called_once()
    update: TwinUpdate = mock_connector.update_node.call_args[0][0]

    assert len(update.derived_facts) == 1
    assert update.derived_facts[0].predicate == "HAS_STATE"
    assert update.derived_facts[0].object == "Acidic"


def test_connector_failure_handling(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    mock_connector.update_node.side_effect = RuntimeError("Connection lost")

    synced = twin_syncer.sync_state("Bio-1", "ph", 7.0, "timestamp")

    # Should handle exception and return False
    assert synced is False


def test_custom_threshold_override(twin_syncer: TwinSyncer) -> None:
    entity_id = "Bio-1"
    ts = "timestamp"

    twin_syncer.sync_state(entity_id, "temp", 100.0, ts)

    # 101.0 is 1% change. Default is 10%. Should throttle.
    assert not twin_syncer.sync_state(entity_id, "temp", 101.0, ts)

    # Override with 0.5% (0.005). Should sync.
    assert twin_syncer.sync_state(entity_id, "temp", 101.0, ts, threshold=0.005)


def test_fact_rule_exception_handling(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    def bad_rule(eid: str, val: float) -> Optional[SemanticFact]:
        raise ValueError("Bad rule")

    twin_syncer.register_fact_rule("ph", bad_rule)

    # Should not crash
    synced = twin_syncer.sync_state("Bio-1", "ph", 7.0, "ts")
    assert synced is True

    update = mock_connector.update_node.call_args[0][0]
    assert len(update.derived_facts) == 0


def test_should_sync_new_property(twin_syncer: TwinSyncer) -> None:
    # Coverage for line 78: if property_name not in self._state_cache[entity_id]:
    entity_id = "Bio-1"
    ts = "ts"

    # Sync first property
    twin_syncer.sync_state(entity_id, "ph", 7.0, ts)

    # Sync second property on same entity
    synced = twin_syncer.sync_state(entity_id, "temp", 37.0, ts)
    assert synced is True


def test_multiple_rules(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    # Coverage for multiple rules and one returning None (line 119 condition)
    def rule_none(eid: str, val: float) -> Optional[SemanticFact]:
        return None

    def rule_ok(eid: str, val: float) -> Optional[SemanticFact]:
        return SemanticFact(subject=eid, predicate="IS", object="OK")

    twin_syncer.register_fact_rule("status", rule_none)
    twin_syncer.register_fact_rule("status", rule_ok)

    twin_syncer.sync_state("Bio-1", "status", 1.0, "ts")

    update = mock_connector.update_node.call_args[0][0]
    # Only one fact should be present
    assert len(update.derived_facts) == 1
    assert update.derived_facts[0].object == "OK"


def test_sync_cache_update_logic(twin_syncer: TwinSyncer) -> None:
    # Coverage for line 119: if entity_id not in self._state_cache:
    # This logic runs AFTER _should_sync returns True.
    # _should_sync already populates _state_cache[entity_id] if it was missing (lines 65-66).
    # So line 119 `if entity_id not in self._state_cache:` is theoretically unreachable
    # unless _should_sync behavior changes or threading race condition (which we don't test here).
    #
    # Let's verify line 65:
    # if entity_id not in self._state_cache:
    #    self._state_cache[entity_id] = {}
    #    return True
    #
    # So when sync_state calls _should_sync, it ensures _state_cache[entity_id] exists.
    # Then sync_state continues to line 119.
    # Thus line 119 IS unreachable in single-threaded context if _should_sync works as expected.
    # This might be defensive coding that is untestable without mocking internal state mid-flight.
    #
    # However, we can trick it by manually clearing cache between _should_sync and the check?
    # No, can't interleave easily.
    #
    # Actually, look closely at line 66: it returns True.
    # So sync_state proceeds.
    # But wait, line 118: `if entity_id not in self._state_cache:`
    # Since line 66 set it, it should be there.
    # BUT, what if `_should_sync` returns True because of the OTHER condition (threshold)?
    # Then `entity_id` is already in cache.
    #
    # So line 119 seems truly redundant defensive code.
    # To cover it, we can mock `_should_sync` to return True but NOT set the cache?

    with pytest.MonkeyPatch.context() as m:
        # We mock _should_sync to return True without touching cache
        # But _should_sync is an instance method.

        def mock_should_sync(entity_id: str, property_name: str, value: float, threshold: float) -> bool:
            # force true but don't init cache if missing
            return True

        m.setattr(twin_syncer, "_should_sync", mock_should_sync)

        # Now call sync_state with a new entity ID
        # It enters sync_state -> calls mock_should_sync -> returns True
        # Then hits line 118: if "NewID" not in cache -> True -> enters 119
        twin_syncer.sync_state("NewID", "prop", 1.0, "ts")

        assert "NewID" in twin_syncer._state_cache
        assert twin_syncer._state_cache["NewID"]["prop"] == 1.0
