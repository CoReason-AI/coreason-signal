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
from unittest.mock import MagicMock

import pytest

from coreason_signal.schemas import TwinUpdate
from coreason_signal.twin.syncer import TwinSyncer


class MockConnector:
    def update_node(self, update: TwinUpdate) -> None:
        pass


@pytest.fixture  # type: ignore[misc]
def mock_connector() -> MagicMock:
    return MagicMock(spec=MockConnector)


@pytest.fixture  # type: ignore[misc]
def twin_syncer(mock_connector: MagicMock) -> TwinSyncer:
    return TwinSyncer(connector=mock_connector, default_sigma_threshold=0.1)


def test_recovery_from_sync_failure(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    """
    Test that if a sync fails, the local cache is NOT updated.
    A subsequent attempt with the same value should NOT be throttled.
    """
    entity_id = "Bio-1"
    prop = "temp"
    val = 37.0
    ts = "ts"

    # 1. First sync works (baseline)
    twin_syncer.sync_state(entity_id, prop, 30.0, ts)
    assert twin_syncer._last_synced_state[entity_id][prop] == 30.0

    # 2. Update to 37.0 fails due to network error
    mock_connector.update_node.side_effect = RuntimeError("Network Down")
    synced = twin_syncer.sync_state(entity_id, prop, val, ts)
    assert synced is False
    # CRITICAL: Cache should STILL reflect 30.0, not 37.0
    assert twin_syncer._last_synced_state[entity_id][prop] == 30.0

    # 3. Retry 37.0 (Network Back Up)
    mock_connector.update_node.side_effect = None
    synced = twin_syncer.sync_state(entity_id, prop, val, ts)
    assert synced is True
    # Now cache should be 37.0
    assert twin_syncer._last_synced_state[entity_id][prop] == 37.0


def test_nan_handling(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    """
    Test that NaN values are always synced and don't break the throttle logic.
    """
    entity_id = "Bio-1"
    prop = "do2"

    # 1. Sync valid value
    twin_syncer.sync_state(entity_id, prop, 50.0, "ts")

    # 2. Sync NaN - Should Sync
    # This hits `_should_sync` line 102: if math.isnan(value): return True
    synced = twin_syncer.sync_state(entity_id, prop, float("nan"), "ts")
    assert synced is True

    # Verify we sent NaN
    update = mock_connector.update_node.call_args[0][0]
    assert math.isnan(update.properties[prop])

    # 3. Sync NaN again - Should Sync
    synced = twin_syncer.sync_state(entity_id, prop, float("nan"), "ts")
    assert synced is True


def test_inf_handling(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    """
    Test Infinity handling.
    """
    entity_id = "Bio-1"
    prop = "rpm"

    twin_syncer.sync_state(entity_id, prop, 100.0, "ts")

    synced = twin_syncer.sync_state(entity_id, prop, float("inf"), "ts")
    assert synced is True

    update = mock_connector.update_node.call_args[0][0]
    assert math.isinf(update.properties[prop])


def test_concurrency_smoke(twin_syncer: TwinSyncer, mock_connector: MagicMock) -> None:
    """
    Smoke test to ensure the Syncer doesn't crash under concurrent load.
    Note: Without a lock, race conditions might cause double writes, but shouldn't crash.
    """
    import concurrent.futures

    def worker(i: int) -> None:
        twin_syncer.sync_state("Bio-1", "concurrent_prop", float(i), "ts")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Should not raise

    # Verify final state exists
    assert "Bio-1" in twin_syncer._last_synced_state
    assert "concurrent_prop" in twin_syncer._last_synced_state["Bio-1"]


def test_transition_from_nan_to_valid(twin_syncer: TwinSyncer) -> None:
    """
    Test recovering from NaN to a valid number.
    """
    entity_id = "Bio-1"
    prop = "ph"

    # Sync NaN
    twin_syncer.sync_state(entity_id, prop, float("nan"), "ts")

    # Sync valid - should pass (NaN -> Number is significant)
    synced = twin_syncer.sync_state(entity_id, prop, 7.0, "ts")
    assert synced is True
    assert twin_syncer._last_synced_state[entity_id][prop] == 7.0


def test_is_significant_change_coverage() -> None:
    """
    Directly test static method _is_significant_change to hit edge cases (lines 80-81).
    """
    # Case 1: New value is NaN -> True (Covered by line 80)
    assert TwinSyncer._is_significant_change(1.0, float("nan"), 0.1) is True

    # Case 2: New value is Inf -> True (Covered by line 80)
    assert TwinSyncer._is_significant_change(1.0, float("inf"), 0.1) is True

    # Case 3: Old value is NaN -> True (Covered by line 82)
    assert TwinSyncer._is_significant_change(float("nan"), 1.0, 0.1) is True
