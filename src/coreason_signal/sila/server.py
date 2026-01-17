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

from sila2.server import SilaServer

from coreason_signal.schemas import DeviceDefinition
from coreason_signal.utils.logger import logger


class SiLAGateway:
    """
    The Protocol Bridge / SiLA 2 Gateway.
    Wraps the SiLA 2 Server and dynamically loads features based on DeviceDefinition.
    """

    def __init__(
        self,
        device_def: DeviceDefinition,
        arrow_flight_port: int = 50055,
        server_instance: Optional[SilaServer] = None,
    ):
        """
        Initialize the SiLA Gateway.

        Args:
            device_def: The DeviceDefinition configuration.
            arrow_flight_port: The dedicated port for the sidecar Arrow Flight server.
                               (Managed separately, stored here for discovery/metadata).
            server_instance: Optional injected SiLAServer instance for testing.
        """
        self.device_def = device_def
        self.arrow_flight_port = arrow_flight_port

        # Parse endpoint to extract host and port for SiLA
        # HttpUrl in Pydantic v2 has .host and .port attributes (or .host_str for ipv6)
        self.host = self.device_def.endpoint.host
        self.port = self.device_def.endpoint.port or 50052

        logger.info(
            f"Initializing SiLAGateway for {self.device_def.id} on {self.host}:{self.port}"
        )
        logger.info(f"Sidecar Arrow Flight Port configured at: {self.arrow_flight_port}")

        if server_instance:
            self.server = server_instance
        else:
            # We initialize the real SiLAServer here.
            # In a real implementation, we would pass name, description, etc.
            self.server = SilaServer(
                server_name=self.device_def.id,
                server_description=f"Coreason Signal Gateway for {self.device_def.driver_type}",
                server_type="CoreasonGateway",
                server_version="0.1.0",
                port=self.port,
                # ip=self.host # Note: sila2 lib might use 'ip' or 'address'. verify if possible.
                # Assuming defaults or simple init for now.
            )

        self._load_capabilities()

    def _load_capabilities(self) -> None:
        """
        Dynamically generate and register SiLA features based on device capabilities.
        """
        for capability in self.device_def.capabilities:
            logger.info(f"Dynamically loading capability: {capability}")
            # TODO: In the future, this will map strings to concrete Feature implementations.
            # For now, we log the registration intention.
            # e.g., self.server.add_feature(feature_implementation=...)
            pass

    def start(self) -> None:
        """
        Start the SiLA 2 Server.
        """
        logger.info("Starting SiLAGateway...")
        # Note: server.run() is usually blocking.
        # We wrap it or expect the caller to handle threading if needed.
        self.server.run(block=False)

    def stop(self) -> None:
        """
        Stop the SiLA 2 Server.
        """
        logger.info("Stopping SiLAGateway...")
        if hasattr(self.server, "stop"):
            self.server.stop()
