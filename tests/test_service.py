# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import time
from typing import Dict, Generator
from unittest.mock import MagicMock, patch

import anyio
import pytest

from coreason_signal.main import _shutdown_handler, main
from coreason_signal.service import Service, ServiceAsync


@pytest.fixture  # type: ignore[misc]
def mock_components() -> Generator[Dict[str, MagicMock], None, None]:
    """Mock all heavy external components and yield them."""
    with (
        patch("coreason_signal.service.LocalVectorStore"),
        patch("coreason_signal.service.ReflexEngine"),
        patch("coreason_signal.service.SiLAGateway") as mock_gateway,
        patch("coreason_signal.service.SignalFlightServer") as mock_flight,
    ):
        # Mock instance to simulate blocking behavior for thread testing
        mock_gateway.return_value.start = MagicMock(side_effect=lambda: time.sleep(0.1))
        mock_gateway.return_value.stop = MagicMock()
        mock_flight.return_value.serve = MagicMock(side_effect=lambda: time.sleep(0.1))
        mock_flight.return_value.shutdown = MagicMock()

        yield {"gateway": mock_gateway, "flight": mock_flight}


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_async_setup(mock_components: Dict[str, MagicMock]) -> None:
    """Test ServiceAsync setup."""
    service = ServiceAsync()
    await service.setup()

    assert service.gateway is not None
    assert service.reflex_engine is not None
    assert service.flight_server is not None


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_async_lifecycle(mock_components: Dict[str, MagicMock]) -> None:
    """Test ServiceAsync context manager lifecycle."""
    async with ServiceAsync() as service:
        assert service.gateway is not None
        assert service.flight_server is not None

        # We can also call start
        await service.start()

        # Verify threads started and still alive (due to sleep side effect)
        assert service._gateway_thread is not None
        assert service._gateway_thread.is_alive()
        assert service._flight_thread is not None
        assert service._flight_thread.is_alive()

    # Verify shutdown was called
    mock_components["gateway"].return_value.stop.assert_called()
    mock_components["flight"].return_value.shutdown.assert_called()


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_async_run_forever_cancellation(mock_components: Dict[str, MagicMock]) -> None:
    """Test ServiceAsync run_forever cancellation."""
    service = ServiceAsync()
    await service.setup()

    # Run run_forever in a task and cancel it
    async with anyio.create_task_group() as tg:
        tg.start_soon(service.run_forever)
        await anyio.sleep(0.1)
        tg.cancel_scope.cancel()

    # Should have shut down
    mock_components["gateway"].return_value.stop.assert_called()


def test_service_sync_facade(mock_components: Dict[str, MagicMock]) -> None:
    """Test Service sync facade."""
    service = Service()

    with service:
        # Check if underlying async service is set up
        assert service._async_service.gateway is not None

        # Test start
        service.start()
        assert service._async_service._gateway_thread is not None
        assert service._async_service._gateway_thread.is_alive()

    # Verify shutdown
    mock_components["gateway"].return_value.stop.assert_called()
    mock_components["flight"].return_value.shutdown.assert_called()


def test_service_sync_run_forever(mock_components: Dict[str, MagicMock]) -> None:
    """Test Service.run_forever handles KeyboardInterrupt."""
    service = Service()

    with patch.object(service._async_service, "run_forever", new_callable=MagicMock) as mock_run:
        # Simulate KeyboardInterrupt
        mock_run.side_effect = KeyboardInterrupt()
        service.run_forever()


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_uninitialized_error() -> None:
    """Test that starting without setup raises RuntimeError."""
    service = ServiceAsync()
    with pytest.raises(RuntimeError, match="Service not initialized"):
        await service.start()


def test_main_entry_point(mock_components: Dict[str, MagicMock]) -> None:
    """Test the main entry point."""
    # Mock Service to verify it's used
    with patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        # mock_instance is a Context Manager
        mock_instance.__enter__.return_value = mock_instance

        # Simulate run_forever raising KeyboardInterrupt to exit clean
        mock_instance.run_forever.side_effect = KeyboardInterrupt

        main()

        mock_instance.run_forever.assert_called()


def test_main_exception(mock_components: Dict[str, MagicMock]) -> None:
    """Test main entry point exception handling."""
    with patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance

        # Simulate generic exception
        mock_instance.run_forever.side_effect = Exception("Boom")

        with pytest.raises(Exception, match="Boom"):
            main()


def test_shutdown_handler() -> None:
    """Test the isolated shutdown handler."""
    with pytest.raises(KeyboardInterrupt):
        _shutdown_handler(15, None)
