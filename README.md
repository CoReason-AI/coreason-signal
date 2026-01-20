# coreason-signal

**The Edge Intelligence Gateway for the CoReason Ecosystem.**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI Status](https://github.com/CoReason-AI/coreason_signal/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_signal/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-product_requirements-blue)](docs/product_requirements.md)

**coreason-signal** transforms laboratory instruments from passive data loggers into active participants. It acts as the "nervous system" of the Self-Driving Lab, bridging the gap between cloud-based reasoning and physical hardware.

## Features

- **Edge Agentic AI (Micro-Cortex):** Deploys Micro-LLMs locally to parse error logs, query local SOPs via RAG, and trigger autonomous recovery actions (e.g., retrying failed aspirations).
- **Soft Sensing (Virtual Metrology):** Uses Physics-Informed Neural Networks (PINNs) to infer invisible biological states (like cell viability) from real-time physical sensor data.
- **Protocol Polyglot:** Natively supports **SiLA 2** (Standard in Lab Automation) while wrapping legacy serial and analog instruments (via Computer Vision) into clean microservices.
- **Live Digital Twin:** Syncs physical state to the CoReason Knowledge Graph with sub-second latency using delta updates.
- **High-Throughput Streaming:** Leverages **Apache Arrow Flight** for efficient transmission of high-frequency waveform data.

For detailed requirements and architecture, see [Product Requirements](docs/product_requirements.md).

## Installation

```bash
pip install coreason-signal
```

## Usage

To start the Edge Intelligence Gateway:

```python
from coreason_signal.main import Application
import signal

# Initialize the application
app = Application()

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, app.shutdown)
signal.signal(signal.SIGTERM, app.shutdown)

# Setup and run
try:
    app.setup()
    app.run()
except Exception as e:
    print(f"Fatal error: {e}")
    app.shutdown()
```

Or via the command line interface (CLI):

```bash
poetry run start
```

## License

This project is licensed under the **Prosperity Public License 3.0**.
See [LICENSE](LICENSE) for details.
