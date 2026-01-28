# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Optional, Any

from sila2.server import SilaServer
from coreason_identity.models import UserContext

from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.features import FeatureRegistry
from coreason_signal.utils.logger import logger

# Default ports as per PRD/Architecture
DEFAULT_SILA_PORT = 50052
DEFAULT_ARROW_FLIGHT_PORT = 50055


class SiLAGateway:
    """The Protocol Bridge / SiLA 2 Gateway.

    Wraps the SiLA 2 Server and dynamically loads features based on DeviceDefinition.
    It acts as the primary interface for instrument control.

    Attributes:
        device_def (DeviceDefinition): The configuration for the device.
        arrow_flight_port (int): Port for the associated Arrow Flight data stream.
        server (SilaServer): The underlying SiLA 2 Server instance.
    """

    def __init__(
        self,
        device_def: DeviceDefinition,
        arrow_flight_port: int = DEFAULT_ARROW_FLIGHT_PORT,
        server_instance: Optional[SilaServer] = None,
    ):
        """Initialize the SiLA Gateway.

        Args:
            device_def (DeviceDefinition): The DeviceDefinition configuration.
            arrow_flight_port (int): The dedicated port for the sidecar Arrow Flight server.
                                     (Managed separately, stored here for discovery/metadata).
            server_instance (Optional[SilaServer]): Optional injected SiLAServer instance for testing.
        """
        self.device_def = device_def
        self.arrow_flight_port = arrow_flight_port

        # Parse endpoint to extract host and port for SiLA
        # HttpUrl in Pydantic v2 has .host and .port attributes (or .host_str for ipv6)
        self.host = self.device_def.endpoint.host
        self.port = self.device_def.endpoint.port or DEFAULT_SILA_PORT

        logger.info(f"Initializing SiLAGateway for {self.device_def.id} on {self.host}:{self.port}")
        logger.info(f"Sidecar Arrow Flight Port configured at: {self.arrow_flight_port}")

        if server_instance:
            self.server = server_instance
        else:
            self.server = SilaServer(
                server_name=self.device_def.id,
                server_description=f"Coreason Signal Gateway for {self.device_def.driver_type}",
                server_type="CoreasonGateway",
                server_version="0.3.0",
                server_vendor_url="https://coreason.ai",
            )

        self._load_capabilities()

    def _load_capabilities(self) -> None:
        """Dynamically generate and register SiLA features based on device capabilities."""
        for capability in self.device_def.capabilities:
            logger.info(f"Dynamically loading capability: {capability}")
            try:
                # 1. Create Feature Definition
                feature_def = FeatureRegistry.create_feature(capability)

                # 2. Create Implementation
                impl = FeatureRegistry.create_implementation(self.server, capability)

                # 3. Register with Server
                self.server.set_feature_implementation(feature_def, impl)
                logger.info(f"Successfully loaded capability: {capability}")
            except Exception as e:
                logger.error(f"Failed to load capability {capability}: {e}")

    def start(self) -> None:
        """Start the SiLA 2 Server.

        This method starts the underlying SiLA server on the configured address and port.
        """
        logger.info("Starting SiLAGateway...")
        # Note: server.run() is usually blocking.
        # We wrap it or expect the caller to handle threading if needed.
        # SilaServer.start(address, port, certificate=None, private_key=None, ...)
        # We assume host is a valid address string (e.g. 0.0.0.0 or ip)
        self.server.start(address="0.0.0.0", port=self.port)

    def stop(self) -> None:
        """Stop the SiLA 2 Server."""
        logger.info("Stopping SiLAGateway...")
        if hasattr(self.server, "stop"):
            self.server.stop()

    def handle_request(self, payload: Any, context: UserContext) -> None:
        """Handle an arbitrary request with identity context.

        Args:
            payload (Any): The request payload.
            context (UserContext): The identity context.
        """
        if context is None:
            raise ValueError("UserContext is required.")

        logger.info(
            "Handling SiLA request",
            user_id=context.user_id.get_secret_value(),
            payload_type=type(payload).__name__,
        )
