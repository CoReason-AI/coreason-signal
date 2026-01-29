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
from coreason_identity.models import UserContext
from coreason_signal.schemas import DeviceDefinition
from coreason_signal.sila.server import SiLAGateway
from pydantic import HttpUrl


@pytest.fixture
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
        # assert call_kwargs["port"] == 50052  # Removed as SilaServer takes port in start()

        # Verify internal state
        assert gateway.arrow_flight_port == 50055  # Default
        assert gateway.host == "127.0.0.1"


def test_sila_gateway_dynamic_capabilities(mock_device_def: DeviceDefinition) -> None:
    """Test that capabilities are processed during initialization."""
    with patch("coreason_signal.sila.server.SilaServer") as MockServer:
        # We also need to patch FeatureRegistry since it might fail if sila2 checks types strictly
        # or if we want to verify calls.
        with patch("coreason_signal.sila.server.FeatureRegistry") as MockRegistry:
            # Setup mocks
            mock_feature_def = MagicMock()
            mock_impl = MagicMock()
            MockRegistry.create_feature.return_value = mock_feature_def
            MockRegistry.create_implementation.return_value = mock_impl

            # Setup Server instance mock
            server_instance = MockServer.return_value

            SiLAGateway(device_def=mock_device_def)

            # Verify FeatureRegistry was called for each capability
            assert MockRegistry.create_feature.call_count == 2
            MockRegistry.create_feature.assert_any_call("Transfer")
            MockRegistry.create_feature.assert_any_call("Mix")

            # Verify set_feature_implementation was called
            assert server_instance.set_feature_implementation.call_count == 2
            server_instance.set_feature_implementation.assert_any_call(mock_feature_def, mock_impl)


def test_sila_gateway_capability_loading_error(mock_device_def: DeviceDefinition) -> None:
    """Test error handling during capability loading."""
    with patch("coreason_signal.sila.server.SilaServer"):
        with patch("coreason_signal.sila.server.FeatureRegistry") as MockRegistry:
            # Simulate error on first call
            MockRegistry.create_feature.side_effect = RuntimeError("Invalid Feature")

            SiLAGateway(device_def=mock_device_def)

            # Should not crash, but log error (verified by coverage)
            # We can verify it continued to try or stopped depending on logic.
            # Logic loop continues.
            assert MockRegistry.create_feature.call_count == 2


def test_sila_gateway_start_stop(mock_device_def: DeviceDefinition) -> None:
    """Test start and stop methods delegate to the server."""
    mock_server_instance = MagicMock()
    # We need to mock FeatureRegistry to avoid side effects during init inside Gateway if we were not passing instance
    # But here we pass server_instance, so we just need to ensure _load_capabilities doesn't crash
    # Real _load_capabilities will run.
    # We should patch FeatureRegistry to mock the actual Feature creation logic to avoid dependency on sila2 internals
    # in unit test

    with patch("coreason_signal.sila.server.FeatureRegistry"):
        gateway = SiLAGateway(device_def=mock_device_def, server_instance=mock_server_instance)

        gateway.start()
        # Updated to reflect new start signature: address="0.0.0.0", port=self.port
        mock_server_instance.start.assert_called_once_with(address="0.0.0.0", port=50052)

        gateway.stop()
        mock_server_instance.stop.assert_called_once()


def test_sila_gateway_custom_arrow_port(mock_device_def: DeviceDefinition) -> None:
    """Test configuration of a custom Arrow Flight sidecar port."""
    with patch("coreason_signal.sila.server.SilaServer"):
        with patch("coreason_signal.sila.server.FeatureRegistry"):
            gateway = SiLAGateway(device_def=mock_device_def, arrow_flight_port=9999)
            assert gateway.arrow_flight_port == 9999


def test_sila_gateway_handle_request(mock_device_def: DeviceDefinition, user_context: UserContext) -> None:
    with patch("coreason_signal.sila.server.SilaServer"), patch("coreason_signal.sila.server.FeatureRegistry"):
        gateway = SiLAGateway(device_def=mock_device_def)
        payload = {"cmd": "START"}
        gateway.handle_request(payload, user_context)
        # Verify no error


def test_sila_gateway_handle_request_validation(mock_device_def: DeviceDefinition) -> None:
    with patch("coreason_signal.sila.server.SilaServer"), patch("coreason_signal.sila.server.FeatureRegistry"):
        gateway = SiLAGateway(device_def=mock_device_def)
        with pytest.raises(ValueError, match="UserContext is required"):
            gateway.handle_request({}, None)  # type: ignore[arg-type]
