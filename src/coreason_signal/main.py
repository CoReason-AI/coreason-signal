# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

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
from coreason_signal.utils.logger import logger


class Application:
    """
    Main application orchestrator for Coreason Signal.
    Manages lifecycle of Edge Agent, SiLA Gateway, and other engines.
    """

    def __init__(self) -> None:
        self.shutdown_event = threading.Event()
        self.gateway: Optional[SiLAGateway] = None

    def setup(self) -> None:
        """Initialize all components."""
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
        logger.info("Initialization complete.")

    def run(self) -> None:
        """Start all services and block until shutdown."""
        if not self.gateway:
            raise RuntimeError("Application not initialized. Call setup() first.")

        # Start SiLA Server (it runs in its own thread/process usually, or we wrap it)
        # The wrapper currently calls server.start() which might be blocking depending on sila2 impl.
        # We assume for this orchestrator we want to control the main loop.
        # If SiLAGateway.start() is blocking, we should run it in a thread.
        # Based on previous review, SiLAGateway.start delegates to server.start().

        logger.info("Starting services...")

        # Run Gateway in a separate thread to allow main thread to handle signals
        gateway_thread = threading.Thread(target=self.gateway.start, daemon=True)
        gateway_thread.start()

        logger.info(f"Coreason Signal running on port {settings.SILA_PORT}")

        # Main Loop / Wait for shutdown
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
            self.shutdown()

    def shutdown(self, signum: Optional[int] = None, frame: Optional[FrameType] = None) -> None:
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received. Stopping services...")
        self.shutdown_event.set()
        if self.gateway:
            self.gateway.stop()
        logger.info("Services stopped. Exiting.")


def main() -> None:
    """Entry point."""
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


if __name__ == "__main__":
    main()
