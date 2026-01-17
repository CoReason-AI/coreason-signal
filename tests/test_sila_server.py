# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from unittest.mock import MagicMock, patch

import pytest
from pydantic import HttpUrl

from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.server import SiLAGateway


@pytest.fixture  # type: ignore[misc]
def mock_device_def() -> DeviceDefinition:
    return DeviceDefinition(
        id="TestInstrument",
        driver_type="SiLA2",
        endpoint=HttpUrl("http://127.0.0.1:50052"),
        capabilities=["Transfer", "Mix"],
        edge_agent_model="test_model.onnx",
        allowed_reflexes=["PAUSE"],
    )


def test_sila_gateway_initialization(mock_device_def: DeviceDefinition) -> None:
    """Test that SiLAGateway initializes correctly with the given definition."""
    with patch("coreason_signal.sila.server.SilaServer") as MockServer:
        gateway = SiLAGateway(device_def=mock_device_def)

        # Verify Server was initialized with correct params
        MockServer.assert_called_once()
        call_kwargs = MockServer.call_args[1]
        assert call_kwargs["server_name"] == "TestInstrument"
        assert call_kwargs["port"] == 50052

        # Verify internal state
        assert gateway.arrow_flight_port == 50055  # Default
        assert gateway.host == "127.0.0.1"


def test_sila_gateway_dynamic_capabilities(mock_device_def: DeviceDefinition) -> None:
    """Test that capabilities are processed during initialization."""
    with patch("coreason_signal.sila.server.SilaServer"):
        gateway = SiLAGateway(device_def=mock_device_def)

        # In the current stub implementation, we just expect it to not crash
        # and log the capabilities. We can check if _load_capabilities ran.
        # Since we can't easily spy on internal methods without setup,
        # we rely on the fact that if it crashed, this test would fail.
        assert gateway.server is not None


def test_sila_gateway_start_stop(mock_device_def: DeviceDefinition) -> None:
    """Test start and stop methods delegate to the server."""
    mock_server_instance = MagicMock()
    gateway = SiLAGateway(device_def=mock_device_def, server_instance=mock_server_instance)

    gateway.start()
    mock_server_instance.run.assert_called_once_with(block=False)

    gateway.stop()
    mock_server_instance.stop.assert_called_once()


def test_sila_gateway_custom_arrow_port(mock_device_def: DeviceDefinition) -> None:
    """Test configuration of a custom Arrow Flight sidecar port."""
    with patch("coreason_signal.sila.server.SilaServer"):
        gateway = SiLAGateway(device_def=mock_device_def, arrow_flight_port=9999)
        assert gateway.arrow_flight_port == 9999
