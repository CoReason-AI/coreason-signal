# System Requirements

Coreason Signal is designed to run on edge computing devices (e.g., NVIDIA Jetson, Industrial PCs) as well as standard server environments.

## Software Stack

### Core Dependencies
- **Python:** 3.12+
- **SiLA 2:** For standard laboratory automation connectivity (`sila2`).
- **Apache Arrow Flight:** For high-performance data streaming (`pyarrow`).
- **ONNX Runtime:** For executing Soft Sensor PINNs (`onnxruntime`).
- **LanceDB:** For local vector storage and RAG (`lancedb`).

### Management API (New in v0.4.0)
The REST Control Plane is built on a modern ASGI stack:
- **FastAPI:** High-performance web framework for building APIs.
- **Uvicorn:** ASGI web server implementation.

## Hardware Requirements

- **CPU:** 4+ Cores recommended for concurrent SiLA and Flight handling.
- **RAM:** 8GB+ (16GB recommended for larger vector stores and buffers).
- **Network:** Gigabit Ethernet recommended for Arrow Flight streams.
- **Storage:** NVMe SSD recommended for low-latency Vector Store access.

## Environment Variables

Configure the service using the following environment variables (or `.env` file):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SIGNAL_LOG_LEVEL` | Logging verbosity | `INFO` |
| `SIGNAL_SILA_PORT` | Port for SiLA 2 Server | `50052` |
| `SIGNAL_ARROW_FLIGHT_PORT` | Port for Arrow Flight Server | `50055` |
| `SIGNAL_REFLEX_TIMEOUT` | Max decision time for Reflex Engine | `0.2` (seconds) |
