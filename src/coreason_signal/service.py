# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

"""
Core service logic for Coreason Signal, implementing the Async-Native with Sync Facade pattern.
"""

import contextlib
import threading
from types import TracebackType
from typing import Optional

import anyio
import httpx

from coreason_signal.config import settings
from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.server import SiLAGateway
from coreason_signal.streaming.flight_server import SignalFlightServer
from coreason_signal.utils.logger import logger


class ServiceAsync:
    """Async-native core service for Coreason Signal.

    Handles the lifecycle of the Edge Agent, SiLA Gateway, and other engines.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        """Initialize the ServiceAsync instance.

        Args:
            client (Optional[httpx.AsyncClient]): An optional external HTTP client.
                                                  If not provided, one will be created.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

        self.gateway: Optional[SiLAGateway] = None
        self.flight_server: Optional[SignalFlightServer] = None
        self.reflex_engine: Optional[ReflexEngine] = None

        # Threads for legacy blocking servers
        self._gateway_thread: Optional[threading.Thread] = None
        self._flight_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    async def __aenter__(self) -> "ServiceAsync":
        """Async context manager entry. Initializes resources."""
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit. Cleans up resources."""
        await self.shutdown()
        if self._internal_client:
            await self._client.aclose()

    async def setup(self) -> None:
        """Initialize all components of the application asynchronously."""
        logger.info("Initializing Coreason Signal (Async)...")

        # 1. Initialize RAG / Vector Store
        # Note: LocalVectorStore currently looks synchronous in its init.
        # If it has async init methods, they should be called here.
        # Assuming synchronous init is fine for now as it loads DB.
        # We wrap it in to_thread if it's blocking IO.
        vector_store = await anyio.to_thread.run_sync(
            lambda: LocalVectorStore(db_path=settings.VECTOR_STORE_PATH, embedding_model_name=settings.EMBEDDING_MODEL)
        )

        # 2. Initialize Reflex Engine
        # ReflexEngine init is also sync.
        self.reflex_engine = ReflexEngine(vector_store=vector_store, decision_timeout=settings.REFLEX_TIMEOUT)

        # 3. Load Device Definition
        device_def = DeviceDefinition(
            id="Coreason-Edge-Gateway",
            driver_type="SiLA2",
            endpoint=f"http://0.0.0.0:{settings.SILA_PORT}",
            capabilities=["EdgeAgent"],
            edge_agent_model="default",
            allowed_reflexes=["PAUSE", "NOTIFY"],
        )

        # 4. Initialize SiLA Gateway
        # SiLAGateway init involves loading capabilities, which might be IO bound.
        self.gateway = await anyio.to_thread.run_sync(
            lambda: SiLAGateway(device_def=device_def, arrow_flight_port=settings.ARROW_FLIGHT_PORT)
        )

        # 5. Initialize Arrow Flight Server
        self.flight_server = SignalFlightServer(port=settings.ARROW_FLIGHT_PORT)

        logger.info("Initialization complete.")

    async def start(self) -> None:
        """Start services.

        Since SiLA and FlightServer are blocking servers, we run them in separate threads
        managed by this async service.
        """
        if not self.gateway or not self.flight_server:
            raise RuntimeError("Service not initialized. Call setup() first.")

        logger.info("Starting services...")

        self._shutdown_event.clear()

        # Run Gateway in a separate thread
        # We keep the thread reference to join later if needed,
        # though these servers are designed to run forever until stopped.
        self._gateway_thread = threading.Thread(target=self.gateway.start, daemon=True)
        self._gateway_thread.start()

        # Run Flight Server in a separate thread
        self._flight_thread = threading.Thread(target=self.flight_server.serve, daemon=True)
        self._flight_thread.start()

        logger.info(f"Coreason Signal running: SiLA@{settings.SILA_PORT}, Flight@{settings.ARROW_FLIGHT_PORT}")

    async def run_forever(self) -> None:
        """Run the service until a cancellation signal is received."""
        await self.start()
        try:
            # Sleep forever effectively, but allow cancellation
            while not self._shutdown_event.is_set():
                await anyio.sleep(1.0)
        except anyio.get_cancelled_exc_class():
            logger.info("Service cancelled.")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown services."""
        logger.info("Shutdown signal received. Stopping services...")
        self._shutdown_event.set()

        # Ensure shutdown proceeds even if cancelled
        with anyio.CancelScope(shield=True):
            if self.gateway:
                # gateway.stop() might be blocking
                await anyio.to_thread.run_sync(self.gateway.stop)

            if self.flight_server:
                # flight_server.shutdown() might be blocking
                await anyio.to_thread.run_sync(self.flight_server.shutdown)

        logger.info("Services stopped.")


class Service:
    """Synchronous facade for ServiceAsync."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        """Initialize the facade.

        Args:
            client (Optional[httpx.AsyncClient]): Optional async client to pass to the core.
        """
        self._async_service = ServiceAsync(client=client)
        self._exit_stack: Optional[contextlib.ExitStack] = None

    def __enter__(self) -> "Service":
        """Sync context manager entry."""
        # We use anyio.run to execute the async setup
        # However, __enter__ is synchronous. We can't keep an event loop running
        # across __enter__ and __exit__ easily without a background thread
        # or just running setup here and cleanup in exit.
        # But wait, the task says:
        # "def __enter__(self): return self"
        # "def __exit__(self, *args): anyio.run(self._async.__aexit__, *args)"
        # This implies that the loop is started and stopped per method call OR
        # we rely on methods inside to be wrapped in anyio.run.

        # If we use `async with ServiceAsync()` inside `__enter__` it would close immediately.
        # We need to manually call setup.

        # The prompt says:
        # def __enter__(self):
        #     # Start the event loop for the context?
        #     # Actually typical sync wrappers either start a loop in a thread
        #     # OR just use anyio.run for specific calls.

        # If the user does:
        # with Service() as svc:
        #    svc.do_something()

        # We need `setup` to have run.
        anyio.run(self._async_service.setup)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Sync context manager exit."""
        anyio.run(self._async_service.__aexit__, exc_type, exc_val, exc_tb)

    def start(self) -> None:
        """Start the services."""
        anyio.run(self._async_service.start)

    def run_forever(self) -> None:
        """Run the service forever (blocking)."""
        try:
            anyio.run(self._async_service.run_forever)
        except KeyboardInterrupt:
            # anyio.run might re-raise KeyboardInterrupt or handle it.
            # We want to ensure graceful shutdown is triggered by __exit__ or here.
            pass
