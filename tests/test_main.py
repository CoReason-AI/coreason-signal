# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import threading
import time
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from coreason_signal.main import Application, main


@pytest.fixture  # type: ignore[misc]
def mock_components() -> Generator[None, None, None]:
    """Mock all heavy external components."""
    with (
        patch("coreason_signal.main.LocalVectorStore"),
        patch("coreason_signal.main.ReflexEngine"),
        patch("coreason_signal.main.SiLAGateway") as mock_gateway,
    ):
        # Mock instance
        mock_gateway.return_value.start = MagicMock()
        mock_gateway.return_value.stop = MagicMock()
        yield


def test_app_setup(mock_components: None) -> None:
    """Test that setup initializes all required components."""
    app = Application()
    app.setup()

    assert app.gateway is not None
    assert app.reflex_engine is not None


def test_app_run_shutdown(mock_components: None) -> None:
    """Test the run loop and graceful shutdown mechanism."""
    app = Application()
    app.setup()
    assert app.gateway is not None

    # Run application in a separate thread so we can trigger shutdown from test
    run_thread = threading.Thread(target=app.run)
    run_thread.start()

    # Wait briefly to let it start
    time.sleep(0.1)

    # Verify gateway started
    # We ignore type errors here because app.gateway is a Mock at runtime due to patching
    app.gateway.start.assert_called()  # type: ignore[attr-defined]

    # Trigger shutdown
    app.shutdown()

    # Join thread (it should exit after shutdown is set)
    run_thread.join(timeout=2.0)

    assert not run_thread.is_alive()
    app.gateway.stop.assert_called()  # type: ignore[attr-defined]


def test_main_entry_point(mock_components: None) -> None:
    """Test the main function entry point."""
    # Mock Application to avoid running the actual loop
    with patch("coreason_signal.main.Application") as MockApp:
        mock_app_instance = MockApp.return_value

        main()

        MockApp.assert_called()
        mock_app_instance.setup.assert_called()
        mock_app_instance.run.assert_called()


def test_run_without_setup_raises() -> None:
    """Test that running without setup raises RuntimeError."""
    app = Application()
    with pytest.raises(RuntimeError, match="Application not initialized"):
        app.run()


def test_run_keyboard_interrupt(mock_components: None) -> None:
    """Test handling of KeyboardInterrupt during run loop."""
    app = Application()
    app.setup()

    # Mock time.sleep to raise KeyboardInterrupt
    with patch("time.sleep", side_effect=KeyboardInterrupt):
        app.run()

    # Verify shutdown was called
    assert app.shutdown_event.is_set()
    assert app.gateway is not None
    # We ignore type errors here because app.gateway is a Mock at runtime due to patching
    app.gateway.stop.assert_called()  # type: ignore[attr-defined]


def test_main_exception_handling(mock_components: None) -> None:
    """Test that main handles exceptions and ensures shutdown."""
    with patch("coreason_signal.main.Application") as MockApp:
        mock_app_instance = MockApp.return_value

        # Make setup succeed but run fail
        mock_app_instance.run.side_effect = RuntimeError("Crash")

        with pytest.raises(RuntimeError, match="Crash"):
            main()

        mock_app_instance.shutdown.assert_called()
