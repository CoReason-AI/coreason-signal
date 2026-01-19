# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

"""Main application entry point for Coreason Signal."""

import signal
import threading
import time
from types import FrameType
from typing import Optional

from coreason_signal.config import settings
from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.edge_agent.vector_store import LocalVectorStore
from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.server import SiLAGateway
from coreason_signal.streaming.flight_server import SignalFlightServer
from coreason_signal.utils.logger import logger


class Application:
    """Main application orchestrator for Coreason Signal.

    Manages the lifecycle of the Edge Agent, SiLA Gateway, and other engines.
    It serves as the central entry point that ties together the instrument control (SiLA),
    data streaming (Arrow Flight), and local intelligence (Reflex Engine).

    Attributes:
        shutdown_event (threading.Event): Event to signal application shutdown.
        gateway (Optional[SiLAGateway]): The SiLA 2 Gateway instance.
        flight_server (Optional[SignalFlightServer]): The Arrow Flight Server instance.
        reflex_engine (Optional[ReflexEngine]): The Reflex Engine for autonomous decision making.
    """

    def __init__(self) -> None:
        """Initializes the Application state."""
        self.shutdown_event = threading.Event()
        self.gateway: Optional[SiLAGateway] = None
        self.flight_server: Optional[SignalFlightServer] = None
        self.reflex_engine: Optional[ReflexEngine] = None

    def setup(self) -> None:
        """Initialize all components of the application.

        This method sets up the:
        1. Local Vector Store for RAG.
        2. Reflex Engine for decision logic.
        3. SiLA Gateway for instrument control.
        4. Arrow Flight Server for data streaming.

        Raises:
            Exception: If any component fails to initialize.
        """
        logger.info("Initializing Coreason Signal...")

        # 1. Initialize RAG / Vector Store
        vector_store = LocalVectorStore(
            db_path=settings.VECTOR_STORE_PATH, embedding_model_name=settings.EMBEDDING_MODEL
        )

        # 2. Initialize Reflex Engine
        self.reflex_engine = ReflexEngine(vector_store=vector_store, decision_timeout=settings.REFLEX_TIMEOUT)

        # 3. Load Device Definition (In a real app, this might come from a file or config)
        # For this atomic unit, we use a default definition consistent with the settings.
        device_def = DeviceDefinition(
            id="Coreason-Edge-Gateway",
            driver_type="SiLA2",
            endpoint=f"http://0.0.0.0:{settings.SILA_PORT}",
            capabilities=["EdgeAgent"],  # Placeholder
            edge_agent_model="default",
            allowed_reflexes=["PAUSE", "NOTIFY"],
        )

        # 4. Initialize SiLA Gateway
        self.gateway = SiLAGateway(device_def=device_def, arrow_flight_port=settings.ARROW_FLIGHT_PORT)

        # 5. Initialize Arrow Flight Server
        self.flight_server = SignalFlightServer(port=settings.ARROW_FLIGHT_PORT)

        logger.info("Initialization complete.")

    def run(self) -> None:
        """Start all services and block until shutdown.

        Launches the SiLA Gateway and Flight Server in separate threads and
        enters a main loop waiting for a shutdown signal.

        Raises:
            RuntimeError: If the application has not been initialized via setup().
        """
        if not self.gateway or not self.flight_server:
            raise RuntimeError("Application not initialized. Call setup() first.")

        logger.info("Starting services...")

        # Run Gateway in a separate thread
        gateway_thread = threading.Thread(target=self.gateway.start, daemon=True)
        gateway_thread.start()

        # Run Flight Server in a separate thread (serve() is blocking)
        flight_thread = threading.Thread(target=self.flight_server.serve, daemon=True)
        flight_thread.start()

        logger.info(f"Coreason Signal running: SiLA@{settings.SILA_PORT}, Flight@{settings.ARROW_FLIGHT_PORT}")

        # Main Loop / Wait for shutdown
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
            self.shutdown()

    def shutdown(self, signum: Optional[int] = None, frame: Optional[FrameType] = None) -> None:
        """Graceful shutdown handler.

        Stops the SiLA Gateway and Flight Server.

        Args:
            signum: The signal number (if called by signal handler).
            frame: The current stack frame (if called by signal handler).
        """
        logger.info("Shutdown signal received. Stopping services...")
        self.shutdown_event.set()
        if self.gateway:
            self.gateway.stop()
        if self.flight_server:
            self.flight_server.shutdown()
        logger.info("Services stopped. Exiting.")


def main() -> None:
    """Entry point for the application.

    Sets up signal handlers and runs the Application.
    """
    app = Application()

    # Register signal handlers
    signal.signal(signal.SIGINT, app.shutdown)
    signal.signal(signal.SIGTERM, app.shutdown)

    try:
        app.setup()
        app.run()
    except Exception as e:
        logger.exception(f"Fatal application error: {e}")
        # Ensure shutdown is called even on error if partially started
        app.shutdown()
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
