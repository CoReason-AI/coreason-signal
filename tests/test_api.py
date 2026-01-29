# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from coreason_signal.api import app, lifespan
from coreason_signal.schemas import DeviceDefinition


@pytest.fixture
def mock_service() -> Generator[MagicMock, None, None]:
    """Fixture to provide a mocked ServiceAsync instance."""
    # We avoid spec=ServiceAsync to prevent auto-creation of MagicMocks for methods
    # that might conflict with our explicit AsyncMock assignment.
    service_mock = MagicMock()

    # Async methods need AsyncMock
    service_mock.setup = AsyncMock(return_value=None)
    service_mock.start = AsyncMock(return_value=None)
    service_mock.shutdown = AsyncMock(return_value=None)

    # Setup attributes required by endpoints
    service_mock.gateway = MagicMock()
    service_mock.gateway.device_def = DeviceDefinition(
        id="test-device",
        driver_type="SiLA2",
        endpoint="http://localhost:50052",
        capabilities=[],
        edge_agent_model="test",
        allowed_reflexes=[],
    )
    service_mock.gateway.port = 50052

    service_mock.flight_server = MagicMock()
    service_mock.flight_server.location = "grpc://localhost:50055"
    service_mock.flight_server.get_latest_data.return_value = []

    service_mock.reflex_engine = MagicMock()
    service_mock.soft_sensor_engine = MagicMock()

    yield service_mock


@pytest.fixture
def client(mock_service: MagicMock) -> Generator[TestClient, None, None]:
    # Inject the mock service into the app state
    app.state.service = mock_service
    with TestClient(app) as client:
        yield client


def test_status_endpoint(client: TestClient) -> None:
    """Test GET /status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["device_id"] == "test-device"
    assert data["sila_port"] == 50052
    assert data["status"] == "active"


def test_status_endpoint_not_initialized(client: TestClient) -> None:
    """Test GET /status when service is not ready."""
    # Temporarily remove service
    app.state.service = None
    response = client.get("/status")
    assert response.status_code == 503


def test_status_endpoint_gateway_not_ready(client: TestClient, mock_service: MagicMock) -> None:
    """Test GET /status when gateway is not ready."""
    # Gateway is None
    mock_service.gateway = None
    response = client.get("/status")
    assert response.status_code == 503

    # Gateway def is None
    mock_service.gateway = MagicMock()
    mock_service.gateway.device_def = None
    response = client.get("/status")
    assert response.status_code == 503


def test_latest_sensors_endpoint(client: TestClient) -> None:
    """Test GET /sensors/latest endpoint."""
    response = client.get("/sensors/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["buffered_batches_count"] == 0
    assert data["batches"] == []


def test_latest_sensors_with_data(client: TestClient, mock_service: MagicMock) -> None:
    """Test GET /sensors/latest with data in buffer."""
    batch_mock = MagicMock()
    batch_mock.num_rows = 10
    batch_mock.num_columns = 5
    batch_mock.schema = "Schema<foo: int32>"
    batch_mock.nbytes = 100

    mock_service.flight_server.get_latest_data.return_value = [batch_mock]

    response = client.get("/sensors/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["buffered_batches_count"] == 1
    assert data["batches"][0]["num_rows"] == 10
    assert data["batches"][0]["schema"] == "Schema<foo: int32>"


def test_latest_sensors_endpoint_error(client: TestClient, mock_service: MagicMock) -> None:
    """Test GET /sensors/latest handles exceptions."""
    mock_service.flight_server.get_latest_data.side_effect = RuntimeError("Flight Error")
    response = client.get("/sensors/latest")
    assert response.status_code == 500
    assert "Flight Error" in response.json()["detail"]


def test_latest_sensors_flight_server_missing(client: TestClient, mock_service: MagicMock) -> None:
    """Test GET /sensors/latest when flight server is missing."""
    mock_service.flight_server = None
    response = client.get("/sensors/latest")
    assert response.status_code == 503


def test_latest_sensors_service_not_initialized(client: TestClient) -> None:
    """Test GET /sensors/latest when service is not initialized."""
    app.state.service = None
    response = client.get("/sensors/latest")
    assert response.status_code == 503


def test_trigger_reflex_endpoint(client: TestClient, mock_service: MagicMock) -> None:
    """Test POST /reflex/trigger endpoint."""
    payload = {"action": "TEST_ACTION", "parameters": {"speed": 100}, "reasoning": "Manual Trigger"}
    response = client.post("/reflex/trigger", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "triggered"
    mock_service.reflex_engine.trigger.assert_called_once()


def test_trigger_reflex_not_available(client: TestClient, mock_service: MagicMock) -> None:
    """Test POST /reflex/trigger when engine is missing."""
    mock_service.reflex_engine = None
    payload = {"action": "TEST", "parameters": {}, "reasoning": "test"}
    response = client.post("/reflex/trigger", json=payload)
    assert response.status_code == 503


def test_update_constraints_endpoint(client: TestClient, mock_service: MagicMock) -> None:
    """Test PUT /soft-sensor/constraints endpoint."""
    constraints = {"min_temp": 10.0, "max_temp": 90.0}
    response = client.put("/soft-sensor/constraints", json=constraints)
    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    mock_service.soft_sensor_engine.update_constraints.assert_called_with(constraints)


def test_update_constraints_not_available(client: TestClient, mock_service: MagicMock) -> None:
    """Test PUT /soft-sensor/constraints when engine is missing."""
    mock_service.soft_sensor_engine = None
    constraints = {"min_temp": 10.0}
    response = client.put("/soft-sensor/constraints", json=constraints)
    assert response.status_code == 503


def test_update_constraints_error(client: TestClient, mock_service: MagicMock) -> None:
    """Test PUT /soft-sensor/constraints handles exceptions."""
    mock_service.soft_sensor_engine.update_constraints.side_effect = ValueError("Invalid Constraint")
    constraints = {"min_temp": 10.0}
    response = client.put("/soft-sensor/constraints", json=constraints)
    assert response.status_code == 400
    assert "Invalid Constraint" in response.json()["detail"]


def test_update_constraints_service_not_initialized(client: TestClient) -> None:
    """Test PUT /soft-sensor/constraints when service is None."""
    app.state.service = None
    constraints = {"min_temp": 10.0}
    response = client.put("/soft-sensor/constraints", json=constraints)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_lifespan_no_service() -> None:
    """Test lifespan startup when service is missing in state."""
    mock_app = MagicMock(spec=app)
    mock_app.state.service = None

    # Test that it yields without crashing and logs error
    async with lifespan(mock_app):
        pass
    # No assertion needed other than it didn't raise
