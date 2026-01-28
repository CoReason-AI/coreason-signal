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
from typing import Dict, Generator, cast
from unittest.mock import MagicMock, patch

import anyio
import pytest
from coreason_identity.models import UserContext

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


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_async_run_forever_context(
    mock_components: Dict[str, MagicMock],
    user_context: UserContext,
) -> None:
    """Test ServiceAsync run_forever with context."""
    service = ServiceAsync()
    await service.setup()

    # Run run_forever in a task and cancel it
    async with anyio.create_task_group() as tg:
        tg.start_soon(service.run_forever, user_context)
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


def test_service_sync_run_forever(mock_components: Dict[str, MagicMock], user_context: UserContext) -> None:
    """Test Service.run_forever handles KeyboardInterrupt and context."""
    service = Service()

    with patch.object(service._async_service, "run_forever", new_callable=MagicMock) as mock_run:
        # Simulate KeyboardInterrupt
        mock_run.side_effect = KeyboardInterrupt()
        service.run_forever()

        # Test with context
        service.run_forever(context=user_context)
        mock_run.assert_called_with(user_context)


@pytest.mark.asyncio  # type: ignore[misc]
async def test_service_uninitialized_error() -> None:
    """Test that starting without setup raises RuntimeError."""
    service = ServiceAsync()
    with pytest.raises(RuntimeError, match="Service not initialized"):
        await service.start()


def test_main_entry_point_serve(mock_components: Dict[str, MagicMock]) -> None:
    """Test the main entry point with serve command."""
    # Mock Service to verify it's used
    with patch("sys.argv", ["main", "serve"]), patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        # mock_instance is a Context Manager
        mock_instance.__enter__.return_value = mock_instance

        # Simulate run_forever raising KeyboardInterrupt to exit clean
        mock_instance.run_forever.side_effect = KeyboardInterrupt

        main()

        mock_instance.run_forever.assert_called()
        # Verify context passed
        args, kwargs = mock_instance.run_forever.call_args
        assert kwargs["context"].user_id.get_secret_value() == "cli-user"


def test_main_exception(mock_components: Dict[str, MagicMock]) -> None:
    """Test main entry point exception handling."""
    with patch("sys.argv", ["main", "serve"]), patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance

        # Simulate generic exception
        mock_instance.run_forever.side_effect = Exception("Boom")

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_service_ingest_and_query(mock_components: Dict[str, MagicMock], user_context: UserContext) -> None:
    service = Service()
    with service:
        # Test ingest
        # Need a valid LogEvent dict to trigger decide
        data = {
            "id": "evt-1",
            "timestamp": "2023-01-01T00:00:00",
            "level": "INFO",
            "source": "test",
            "message": "test msg"
        }
        service.ingest_signal(data, user_context)

        # Verify decide called
        assert service._async_service.reflex_engine is not None
        cast(MagicMock, service._async_service.reflex_engine).decide.assert_called()

        # Test invalid ingest
        service.ingest_signal({"invalid": "data"}, user_context)
        # Should not crash

        # Test query
        service.query_signals("fail", 3, user_context)
        # ServiceAsync.reflex_engine is mocked in mock_components via patch("coreason_signal.service.ReflexEngine")
        # So service._async_service.reflex_engine is an instance of the mock.
        # We need to access the return value of the class mock, which is assigned to self.reflex_engine
        # The mock_components yields a dict, but doesn't expose reflex_engine mock explicitly, only patches it.
        # However, we can inspect what was set.
        assert service._async_service.reflex_engine is not None
        # Use cast or assume standard mock behavior for dynamic attribute
        cast(MagicMock, service._async_service.reflex_engine)._vector_store.query.assert_called_with("fail", k=3)


def test_service_query_no_engine(mock_components: Dict[str, MagicMock], user_context: UserContext) -> None:
    service = Service()
    # reflex_engine is None by default
    res = service.query_signals("q", 1, user_context)
    assert res == []


def test_service_identity_validation() -> None:
    service = Service()
    with pytest.raises(ValueError, match="UserContext is required"):
        service.ingest_signal({}, None)

    with pytest.raises(ValueError, match="UserContext is required"):
        service.query_signals("q", 1, None)


def test_main_ingest(mock_components: Dict[str, MagicMock]) -> None:
    with (
        patch("sys.argv", ["main", "ingest", '{"key": "value"}']),
        patch("coreason_signal.main.Service") as MockService,
    ):
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance

        main()

        mock_instance.ingest_signal.assert_called_once()
        args, _ = mock_instance.ingest_signal.call_args
        assert args[0] == {"key": "value"}
        assert args[1].user_id.get_secret_value() == "cli-user"


def test_main_query(mock_components: Dict[str, MagicMock]) -> None:
    with patch("sys.argv", ["main", "query", "myquery"]), patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance

        # Mock query return (assuming simple dict for simplicity)
        mock_instance.query_signals.return_value = [{"id": "1"}]

        main()

        mock_instance.query_signals.assert_called_once()
        args, kwargs = mock_instance.query_signals.call_args
        assert args[0] == "myquery"
        assert args[1] == 3  # default top_k


def test_main_ingest_bad_json(mock_components: Dict[str, MagicMock]) -> None:
    with patch("sys.argv", ["main", "ingest", "{bad_json"]), patch("coreason_signal.main.Service"):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1


def test_main_query_raw_results(mock_components: Dict[str, MagicMock]) -> None:
    with patch("sys.argv", ["main", "query", "q"]), patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance
        # Return simple dicts without model_dump
        mock_instance.query_signals.return_value = [{"raw": "data"}]

        # Capture stdout to verify (optional, or just ensure no crash)
        main()


def test_main_query_model_results(mock_components: Dict[str, MagicMock]) -> None:
    with patch("sys.argv", ["main", "query", "q"]), patch("coreason_signal.main.Service") as MockService:
        mock_instance = MockService.return_value
        mock_instance.__enter__.return_value = mock_instance

        # Mock result with model_dump
        mock_res = MagicMock()
        mock_res.model_dump.return_value = {"id": "1", "dumped": True}
        mock_instance.query_signals.return_value = [mock_res]

        main()


def test_shutdown_handler() -> None:
    """Test the isolated shutdown handler."""
    with pytest.raises(KeyboardInterrupt):
        _shutdown_handler(15, None)
