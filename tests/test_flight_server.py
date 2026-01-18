# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import threading
import time
from typing import Generator
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.flight as flight
import pytest

from coreason_signal.streaming.flight_server import SignalFlightServer


@pytest.fixture  # type: ignore[misc]
def flight_server() -> Generator[SignalFlightServer, None, None]:
    """
    Fixture to start and stop the Flight Server.
    """
    server = SignalFlightServer(port=0)  # Bind to any available port
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()

    # Wait for server to start (simple sleep, ideally would check readiness)
    time.sleep(0.5)

    yield server

    server.shutdown()
    server_thread.join(timeout=2.0)


def test_server_init(flight_server: SignalFlightServer) -> None:
    """Test that the server initializes correctly."""
    assert flight_server.port > 0
    assert len(flight_server.get_latest_data()) == 0


def test_do_put_and_get(flight_server: SignalFlightServer) -> None:
    """Test sending data via do_put and retrieving it via do_get and internal method."""
    location = f"grpc://0.0.0.0:{flight_server.port}"
    client = flight.FlightClient(location)

    # Create dummy data
    data = [
        pa.array([1.0, 2.0, 3.0]),
        pa.array([4.0, 5.0, 6.0]),
    ]
    batch = pa.RecordBatch.from_arrays(data, names=["col1", "col2"])

    # Send data (do_put)
    descriptor = flight.FlightDescriptor.for_path("test_stream")
    writer, _ = client.do_put(descriptor, batch.schema)
    writer.write(batch)
    writer.close()

    # Verify internal buffer
    latest = flight_server.get_latest_data()
    assert len(latest) == 1
    assert latest[0].equals(batch)

    # Verify retrieval (do_get)
    ticket = flight.Ticket(b"test_stream")
    reader = client.do_get(ticket)
    table = reader.read_all()
    assert table.to_batches()[0].equals(batch)


def test_buffer_rolling(flight_server: SignalFlightServer) -> None:
    """Test that the buffer respects the maxlen."""
    # Re-init with small buffer for testing (hacky but effective for unit test)
    flight_server._buffer = flight_server._buffer.__class__(maxlen=2)

    location = f"grpc://0.0.0.0:{flight_server.port}"
    client = flight.FlightClient(location)

    batch1 = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    batch2 = pa.RecordBatch.from_arrays([pa.array([2])], names=["a"])
    batch3 = pa.RecordBatch.from_arrays([pa.array([3])], names=["a"])

    descriptor = flight.FlightDescriptor.for_path("stream")
    writer, _ = client.do_put(descriptor, batch1.schema)

    writer.write(batch1)
    writer.write(batch2)
    writer.write(batch3)
    writer.close()

    latest = flight_server.get_latest_data()
    assert len(latest) == 2
    assert latest[0].equals(batch2)
    assert latest[1].equals(batch3)


def test_list_flights_and_info(flight_server: SignalFlightServer) -> None:
    """Test list_flights and get_flight_info."""
    location = f"grpc://0.0.0.0:{flight_server.port}"
    client = flight.FlightClient(location)

    # Empty initially
    flights = list(client.list_flights())
    assert len(flights) == 0

    # Put data
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    descriptor = flight.FlightDescriptor.for_path("stream")
    writer, _ = client.do_put(descriptor, batch.schema)
    writer.write(batch)
    writer.close()

    # Now we should have flights
    flights = list(client.list_flights())
    assert len(flights) == 1
    assert flights[0].schema.equals(batch.schema)

    # Get info
    info = client.get_flight_info(descriptor)
    assert info.schema.equals(batch.schema)


def test_do_get_unavailable(flight_server: SignalFlightServer) -> None:
    """Test do_get raises unavailable if no data."""
    location = f"grpc://0.0.0.0:{flight_server.port}"
    client = flight.FlightClient(location)
    ticket = flight.Ticket(b"stream")

    # Check for both wrapper and underlying error just in case, but FlightUnavailableError is the key
    with pytest.raises((flight.FlightUnavailableError, flight.FlightServerError)):
        client.do_get(ticket).read_all()


def test_do_put_error_handling(flight_server: SignalFlightServer) -> None:
    """Test that do_put handles exceptions from the reader."""
    mock_reader = MagicMock()
    mock_reader.read_chunk.side_effect = RuntimeError("Mock failure")

    context = MagicMock()
    descriptor = flight.FlightDescriptor.for_path("stream")

    with pytest.raises(RuntimeError, match="Mock failure"):
        flight_server.do_put(context, descriptor, mock_reader, MagicMock())


def test_do_put_stop_iteration(flight_server: SignalFlightServer) -> None:
    """Test do_put loop break on StopIteration."""
    mock_reader = MagicMock()
    # First call returns chunk, second raises StopIteration
    mock_chunk = MagicMock()
    mock_chunk.__bool__.return_value = True  # Make it truthy

    mock_reader.read_chunk.side_effect = [(mock_chunk, None), StopIteration()]

    context = MagicMock()
    descriptor = flight.FlightDescriptor.for_path("stream")

    # Should not raise exception
    flight_server.do_put(context, descriptor, mock_reader, MagicMock())

    # Verify we buffered 1 chunk
    assert len(flight_server.get_latest_data()) == 1


# --- Direct Method Invocation Tests for Coverage ---


def test_do_get_direct(flight_server: SignalFlightServer) -> None:
    """Directly call do_get to ensure coverage."""
    # Pre-populate buffer
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    flight_server._buffer.append(batch)

    context = MagicMock()
    ticket = flight.Ticket(b"ticket")

    stream = flight_server.do_get(context, ticket)
    assert stream is not None

    # Verify generator logic directly
    snapshot = [batch]
    gen = flight_server._stream_generator(snapshot)
    assert list(gen) == snapshot


def test_do_get_direct_unavailable(flight_server: SignalFlightServer) -> None:
    """Directly call do_get with empty buffer to ensure coverage."""
    # Buffer is empty by default
    context = MagicMock()
    ticket = flight.Ticket(b"ticket")

    with pytest.raises(flight.FlightUnavailableError):
        flight_server.do_get(context, ticket)


def test_list_flights_direct(flight_server: SignalFlightServer) -> None:
    """Directly call list_flights to ensure coverage."""
    context = MagicMock()
    criteria = b""

    # 1. Empty buffer
    gen = flight_server.list_flights(context, criteria)
    results = list(gen)
    assert len(results) == 0

    # 2. Populated buffer
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    flight_server._buffer.append(batch)

    gen = flight_server.list_flights(context, criteria)
    results = list(gen)
    assert len(results) == 1
    assert isinstance(results[0], flight.FlightInfo)


def test_get_flight_info_direct(flight_server: SignalFlightServer) -> None:
    """Directly call get_flight_info to ensure coverage."""
    context = MagicMock()
    descriptor = flight.FlightDescriptor.for_path("stream")

    # 1. Empty buffer -> Raise
    with pytest.raises(flight.FlightUnavailableError):
        flight_server.get_flight_info(context, descriptor)

    # 2. Populated
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    flight_server._buffer.append(batch)

    info = flight_server.get_flight_info(context, descriptor)
    assert info is not None
