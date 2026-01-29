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
from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.server import SiLAGateway
from pydantic import HttpUrl


@pytest.fixture
def base_device_def() -> DeviceDefinition:
    return DeviceDefinition(
        id="TestInstrument",
        driver_type="SiLA2",
        endpoint=HttpUrl("http://127.0.0.1:50052"),
        capabilities=["Transfer"],
        edge_agent_model="test_model.onnx",
        allowed_reflexes=["PAUSE"],
    )


def test_sila_gateway_ipv6_endpoint(base_device_def: DeviceDefinition) -> None:
    """Test parsing of IPv6 endpoints."""
    # Note: Pydantic HttpUrl handles IPv6 inside brackets
    base_device_def.endpoint = HttpUrl("http://[::1]:50052")

    with patch("coreason_signal.sila.server.SilaServer") as MockServer:
        gateway = SiLAGateway(device_def=base_device_def)

        # Pydantic v2 .host returns the address. For IPv6 it might strip brackets or keep them depending on version.
        # But we mostly care that it didn't crash and extracted a host.
        assert gateway.host is not None
        assert "::1" in str(gateway.host)
        assert gateway.port == 50052

        MockServer.assert_called_once()


def test_sila_gateway_empty_capabilities(base_device_def: DeviceDefinition) -> None:
    """Test initialization with no capabilities."""
    base_device_def.capabilities = []

    with patch("coreason_signal.sila.server.SilaServer"):
        gateway = SiLAGateway(device_def=base_device_def)
        # Should proceed without error
        assert gateway.server is not None


def test_sila_gateway_server_init_failure(base_device_def: DeviceDefinition) -> None:
    """Test handling of underlying server initialization failure (e.g. port conflict)."""
    with patch("coreason_signal.sila.server.SilaServer", side_effect=OSError("Address already in use")):
        with pytest.raises(OSError, match="Address already in use"):
            SiLAGateway(device_def=base_device_def)


def test_sila_gateway_idempotent_stop(base_device_def: DeviceDefinition) -> None:
    """Test that stop() is robust to multiple calls or being called before start."""
    mock_server_instance = MagicMock()
    gateway = SiLAGateway(device_def=base_device_def, server_instance=mock_server_instance)

    # 1. Stop without start
    gateway.stop()
    mock_server_instance.stop.assert_called_once()

    # 2. Stop again
    gateway.stop()
    assert mock_server_instance.stop.call_count == 2
