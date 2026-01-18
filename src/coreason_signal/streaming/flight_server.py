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
from collections import deque
from typing import Any, Dict, Generator, List, Optional

import pyarrow as pa
import pyarrow.flight as flight

from coreason_signal.utils.logger import logger


class SignalFlightServer(flight.FlightServerBase):  # type: ignore[misc]
    """
    Apache Arrow Flight Server for high-frequency sensor data streaming.
    Buffers incoming data for consumption by the Soft Sensor Engine and Twin Syncer.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50055,
        buffer_size: int = 1000,
        verify_client: bool = False,
        root_certificates: Optional[bytes] = None,
        middleware: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Flight Server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            buffer_size: Max number of record batches to keep in memory.
            verify_client: Whether to enable mTLS client verification (default False for internal sidecar).
            root_certificates: PEM encoded root certificates for TLS.
            middleware: Middleware dictionary.
        """
        location = f"grpc://{host}:{port}"
        super().__init__(
            location,
            verify_client=verify_client,
            root_certificates=root_certificates,
            middleware=middleware,
        )
        self.location = location
        self._buffer: deque[pa.RecordBatch] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        logger.info(f"SignalFlightServer initialized at {location} with buffer size {buffer_size}")

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.FlightMetadataReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        """
        Handle incoming data streams.
        Expects a stream of RecordBatches from the client.
        """
        logger.debug(f"Received do_put request: {descriptor.path}")

        # We assume the path indicates the sensor ID or stream topic
        stream_id = descriptor.path[0].decode("utf-8") if descriptor.path else "unknown"

        try:
            # reader is iterable in newer pyarrow versions or use read_chunk with try/except StopIteration
            while True:
                try:
                    chunk, metadata = reader.read_chunk()
                    # chunk is the RecordBatch itself in this context
                    if chunk:
                        with self._lock:
                            self._buffer.append(chunk)
                except StopIteration:
                    break
        except Exception as e:
            logger.error(f"Error handling stream {stream_id}: {e}")
            raise

    def do_get(
        self,
        context: flight.ServerCallContext,
        ticket: flight.Ticket,
    ) -> flight.GeneratorStream:
        """
        Retrieve buffered data.
        """
        key = ticket.ticket.decode("utf-8")
        logger.debug(f"Received do_get request for ticket: {key}")

        # For this implementation, we simply dump the current buffer
        # In a real scenario, we might filter by the ticket key

        # Snapshot the buffer to avoid locking during iteration
        with self._lock:
            snapshot = list(self._buffer)

        if not snapshot:
            # Return empty stream if no data
            # We need a schema even for empty streams, but if we have no data we don't know the schema yet.
            # Ideally, the server should know the schema beforehand or it should be passed in do_put.
            # For now, we handle this gracefully or raise if strictly required.
            # Here we simply return generic empty response or raise if schema unknown.
            # To be safe, we can check if we have any data.
            raise flight.FlightUnavailableError("No data buffered yet")

        schema = snapshot[0].schema
        return flight.GeneratorStream(schema, self._stream_generator(snapshot))

    def _stream_generator(self, snapshot: List[pa.RecordBatch]) -> Generator[pa.RecordBatch, None, None]:
        """
        Internal generator to yield batches from a snapshot.
        """
        for batch in snapshot:
            yield batch

    def list_flights(
        self, context: flight.ServerCallContext, criteria: bytes
    ) -> Generator[flight.FlightInfo, None, None]:
        """
        List available streams.
        """
        # Placeholder: In a real system, we'd track active streams.
        # For now, we yield a single info if data exists.
        with self._lock:
            if self._buffer:
                descriptor = flight.FlightDescriptor.for_path(b"sensor_stream")
                yield self._create_flight_info(descriptor, self._buffer[0].schema)

    def get_flight_info(
        self, context: flight.ServerCallContext, descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        """
        Get info for a specific stream.
        """
        with self._lock:
            if self._buffer:
                return self._create_flight_info(descriptor, self._buffer[0].schema)
            else:
                raise flight.FlightUnavailableError("No data available")

    def _create_flight_info(self, descriptor: flight.FlightDescriptor, schema: pa.Schema) -> flight.FlightInfo:
        """
        Helper to create a FlightInfo object.
        """
        endpoints = [flight.FlightEndpoint(b"sensor_stream", [self.location])]
        return flight.FlightInfo(schema, descriptor, endpoints, -1, -1)

    def get_latest_data(self) -> List[pa.RecordBatch]:
        """
        Internal method for the Soft Sensor / Syncer to access the latest buffer.
        """
        with self._lock:
            return list(self._buffer)
