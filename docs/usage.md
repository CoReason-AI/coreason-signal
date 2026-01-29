# Usage Guide

This guide explains how to use the Coreason Signal Edge Intelligence Gateway, focusing on the Management API and autonomous reflex capabilities.

## Management API (REST)

Coreason Signal exposes a REST Control Plane on port `8000` (default) for remote management, status monitoring, and runtime configuration.

### Endpoint Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Retrieve the current status of the gateway (SiLA/Flight ports, Device ID). |
| `GET` | `/sensors/latest` | Get a metadata summary of the latest buffered sensor data. |
| `POST` | `/reflex/trigger` | Manually trigger a reflex action (e.g., for recovery or testing). |
| `PUT` | `/soft-sensor/constraints` | Update physics constraints for the Soft Sensor Engine at runtime. |

### Examples

#### Check System Status

```bash
curl -X GET "http://localhost:8000/status"
```

**Response:**
```json
{
  "device_id": "Coreason-Edge-Gateway",
  "sila_port": 50052,
  "arrow_flight_port": "grpc://0.0.0.0:50055",
  "status": "active"
}
```

#### Trigger a Reflex Action

You can manually force a reflex action, bypassing the automated decision engine.

```bash
curl -X POST "http://localhost:8000/reflex/trigger" \
     -H "Content-Type: application/json" \
     -d '{
           "action": "ENGAGE_COOLING_LOOP",
           "parameters": {"flow_rate": 50.0},
           "reasoning": "Manual override via API"
         }'
```

#### Update Soft Sensor Constraints

Tighten or loosen the "Glass Box" physics constraints without restarting the service.

```bash
curl -X PUT "http://localhost:8000/soft-sensor/constraints" \
     -H "Content-Type: application/json" \
     -d '{
           "min_temp": 20.0,
           "max_temp": 80.0
         }'
```

## Legacy Interface (CLI)

You can still use the CLI for basic ingestion and querying of the local RAG store.

```bash
# Ingest a log event
poetry run start ingest '{"id": "evt-1", "level": "ERROR", "message": "Temp High", "timestamp": "2023-10-01T12:00:00Z", "source": "sensor-1"}'

# Query similar SOPs
poetry run start query "Overheating protocol"
```
