# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Generator, cast
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
        patch("coreason_signal.main.SignalFlightServer") as mock_flight,
    ):
        # Mock instance
        mock_gateway.return_value.start = MagicMock()
        mock_gateway.return_value.stop = MagicMock()
        mock_flight.return_value.serve = MagicMock()
        mock_flight.return_value.shutdown = MagicMock()
        yield


def test_app_setup(mock_components: None) -> None:
    """Test that setup initializes all required components."""
    app = Application()
    app.setup()

    assert app.gateway is not None
    assert app.reflex_engine is not None
    assert app.flight_server is not None


def test_app_run_shutdown(mock_components: None) -> None:
    """Test the run loop and graceful shutdown mechanism using synchronous execution."""
    app = Application()
    app.setup()
    assert app.gateway is not None
    assert app.flight_server is not None

    # Side effect for sleep: trigger shutdown to break the loop immediately
    def trigger_shutdown(seconds: float) -> None:
        app.shutdown()

    with patch("time.sleep", side_effect=trigger_shutdown):
        # Run synchronously. The sleep call inside run() will trigger shutdown,
        # causing the loop check to fail and exit.
        app.run()

    # Verify gateway started
    cast(MagicMock, app.gateway.start).assert_called()
    cast(MagicMock, app.flight_server.serve).assert_called()

    # Verify shutdown was called and stopped services
    assert app.shutdown_event.is_set()
    cast(MagicMock, app.gateway.stop).assert_called()
    cast(MagicMock, app.flight_server.shutdown).assert_called()


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
    assert app.gateway is not None

    # Mock time.sleep to raise KeyboardInterrupt
    with patch("time.sleep", side_effect=KeyboardInterrupt):
        app.run()

    # Verify shutdown was called
    assert app.shutdown_event.is_set()
    # We ignore type errors here because app.gateway is a Mock at runtime due to patching
    cast(MagicMock, app.gateway.stop).assert_called()


def test_main_exception_handling(mock_components: None) -> None:
    """Test that main handles exceptions and ensures shutdown."""
    with patch("coreason_signal.main.Application") as MockApp:
        mock_app_instance = MockApp.return_value

        # Make setup succeed but run fail
        mock_app_instance.run.side_effect = RuntimeError("Crash")

        with pytest.raises(RuntimeError, match="Crash"):
            main()

        mock_app_instance.shutdown.assert_called()
