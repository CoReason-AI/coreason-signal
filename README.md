# coreason-signal

**The Nervous System for the Self-Driving Lab.**

`coreason-signal` is an edge AI application that orchestrates SiLA 2 compliant instruments, performs soft sensor inference using ONNX models, manages a local vector store for reflex decisions, and streams data via Apache Arrow Flight. It bridges the "Air Gap" between high-level cloud reasoning and the physical reality of the bench, transforming instruments from dumb peripherals into active participants.

[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_signal)
[![CI Status](https://github.com/CoReason-AI/coreason_signal/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_signal/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/CoReason-AI/coreason_signal)

## Installation

```bash
pip install coreason-signal
```

## Features

*   **Universal Connectivity (`sila2`)**: Wraps hardware as **SiLA 2** (Standard in Lab Automation) microservices, creating a uniform, discoverable API surface for the entire lab.
*   **Cognition at the Edge (`lancedb` + `fastembed`)**: Implements a "Reflex Arc" using a serverless vector store for **Local RAG (Retrieval-Augmented Generation)**. It queries Standard Operating Procedures (SOPs) to resolve errors locally without cloud dependency.
*   **Virtual Metrology (`onnxruntime`)**: Runs Physics-Informed Neural Networks (PINNs) via **ONNX** to infer invisible biological insights (like cell viability) from raw sensor data in real-time.
*   **Data Integrity & Streaming (`pydantic` + `pyarrow`)**: Enforces strict schemas for events/reflexes and streams high-frequency waveform data via **Apache Arrow Flight**, preventing control plane congestion.

## Usage

Here is how to initialize and run the application as a gateway:

```python
from coreason_signal.main import Application

# Initialize the Application
# This boots up the SiLA Gateway, Reflex Engine, and Vector Store
app = Application()
app.setup()

# Start the application loop
# It exposes hardware capabilities and runs the "Nervous System" in the background
try:
    app.run()
except KeyboardInterrupt:
    print("Shutting down coreason-signal...")
```

### Reflex Engine Example

```python
from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import LogEvent

# Initialize the engine
# (Usually handled internally by Application, but can be used standalone)
# engine = ReflexEngine(...)

# Simulate an error event
event = LogEvent(
    id="evt_123",
    timestamp="2023-10-27T10:00:00Z",
    level="ERROR",
    source="LiquidHandler-01",
    message="Aspiration timeout: pressure sensor delta < threshold",
    raw_code="ERR_VAC_04"
)

# The engine queries SOPs and returns a reflex
# reflex = engine.decide(event)
# if reflex:
#     print(f"Action: {reflex.action}, Reasoning: {reflex.reasoning}")
```
