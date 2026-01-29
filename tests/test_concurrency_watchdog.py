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
import datetime
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext
from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import AgentReflex, LogEvent


@pytest.fixture
def mock_vector_store() -> MagicMock:
    return MagicMock()


def test_thundering_herd_timeout(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """
    Test the 'Thundering Herd' scenario where multiple concurrent requests
    overwhelm the single-threaded ReflexEngine.

    Scenario:
    - 3 concurrent requests come in.
    - Each request processing takes 600ms (mocked).
    - Timeout is 1000ms (configured).
    - Engine has max_workers=1 (serial processing).

    Expected Timeline:
    - T=0: Req1, Req2, Req3 submitted.
    - T=0: Req1 starts processing.
    - T=600ms: Req1 finishes (Success < 1000ms). Req2 starts.
    - T=1000ms: Req2 times out (has been waiting 600ms + running 400ms). Returns PAUSE.
    - T=1000ms: Req3 times out (has been waiting 1000ms). Returns PAUSE.
    """
    # Initialize with a generous 1.0s timeout to allow for CI overhead
    engine = ReflexEngine(vector_store=mock_vector_store, decision_timeout=1.0)

    # Mock return value for success
    success_reflex = AgentReflex(action="SUCCESS", reasoning="OK")

    # Define a side effect that sleeps 0.6s then returns success
    # 0.6s < 1.0s (Req 1 Success)
    # 0.6s + 0.6s = 1.2s > 1.0s (Req 2 Fail)
    # Margin = 400ms (huge safety buffer for CI)
    def slow_logic(event: LogEvent) -> AgentReflex:
        time.sleep(0.6)
        return success_reflex

    # Patch the internal logic
    with patch.object(engine, "_decide_logic", side_effect=slow_logic):
        # Create 3 events
        events = [
            LogEvent(
                id=f"evt-{i}",
                timestamp=datetime.datetime.now().isoformat(),
                level="ERROR",
                source="test",
                message=f"Request {i}",
            )
            for i in range(3)
        ]

        results: List[Optional[AgentReflex]] = []

        # Use an external executor to simulate concurrent clients
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as client_executor:
            # Launch all 3 simultaneously
            futures = [client_executor.submit(engine.decide, evt, user_context) for evt in events]

            # Gather results
            for f in futures:
                results.append(f.result())

        # Analysis
        # Request 1 should succeed
        assert results[0] is not None
        assert results[0].action == "SUCCESS"

        # Request 2 should be PAUSE (Timeout)
        assert results[1] is not None
        assert results[1].action == "PAUSE"
        assert "Watchdog Timeout" in results[1].reasoning

        # Request 3 should be PAUSE (Timeout)
        assert results[2] is not None
        assert results[2].action == "PAUSE"


def test_recovery_after_congestion(mock_vector_store: MagicMock, user_context: UserContext) -> None:
    """
    Verify that the engine recovers after the congestion clears.
    """
    # Use default 200ms timeout for this test, but adjust logic to be robust
    engine = ReflexEngine(vector_store=mock_vector_store)
    success_reflex = AgentReflex(action="SUCCESS", reasoning="OK")

    # Logic: 1st call is slow (0.3s > 0.2s), subsequent calls are fast
    call_count = 0

    def variable_logic(event: LogEvent) -> AgentReflex:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            time.sleep(0.3)  # Slow
        else:
            time.sleep(0.01)  # Fast
        return success_reflex

    with patch.object(engine, "_decide_logic", side_effect=variable_logic):
        # 1. Slow call
        evt1 = LogEvent(id="1", timestamp="", level="ERROR", source="t", message="m")
        res1 = engine.decide(evt1, user_context)
        assert res1 is not None
        assert res1.action == "PAUSE"

        # Wait for the background task to clear (0.3s sleep + overhead)
        time.sleep(0.4)

        # 2. Fast call
        evt2 = LogEvent(id="2", timestamp="", level="ERROR", source="t", message="m")
        res2 = engine.decide(evt2, user_context)

        # Should succeed now
        assert res2 is not None
        assert res2.action == "SUCCESS"
