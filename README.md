# The Architecture and Utility of coreason-signal

### 1. The Philosophy (The Why)

Scientific laboratories have historically been "read-only" environments for software—passive loggers that record results long after the experiment is finished. The data is often a static file, analyzed only when it's too late to save a failed run. **coreason-signal** inverts this paradigm. It asserts that **"The Lab is a Stream, not a File."**

Designed as the "Nervous System" for the Self-Driving Lab, `coreason-signal` bridges the "Air Gap" between high-level cloud reasoning and the physical reality of the bench. It transforms instruments from dumb peripherals into active participants. By moving intelligence to the edge, it solves the critical latency problem of biological process control. It doesn't just log an error; it *understands* it.

Whether it is a **Reflex Agent** deciding if a vacuum pressure drop requires a retry, or a **Soft Sensor** inferring invisible cell viability from pH trends, `coreason-signal` closes the loop locally. It enables the lab to reason, react, and recover in milliseconds, ensuring that the "Digital Twin" is always in sync with physical reality.

### 2. Under the Hood (The Dependencies & logic)

The package acts as a high-throughput Edge Intelligence Gateway, orchestrated through a carefully selected stack:

*   **Universal Connectivity (`sila2`)**: Rather than writing bespoke drivers for every device, the system wraps all hardware—from legacy serial scales to modern liquid handlers—as **SiLA 2** (Standard in Lab Automation) microservices. This creates a uniform, discoverable API surface for the entire lab.
*   **Cognition at the Edge (`lancedb` + `fastembed`)**: To implement the "Reflex Arc"—the system's ability to make tactical decisions during faults—it embeds a serverless vector store directly into the application process. This enables **Local RAG (Retrieval-Augmented Generation)**, allowing the agent to query Standard Operating Procedures (SOPs) to resolve errors without depending on an internet connection or a central cloud server.
*   **Virtual Metrology (`onnxruntime`)**: The `SoftSensorEngine` leverages the portable ONNX format to run Physics-Informed Neural Networks (PINNs). By running these models locally (potentially accelerated by CUDA or OpenVINO), it transforms high-frequency raw signals into complex biological insights in real-time.
*   **Data Integrity (`pydantic` + `pyarrow`)**: In the chaotic environment of hardware integration, type safety is paramount. **Pydantic** enforces strict schemas for every event and reflex, ensuring that only valid, semantic data propagates to the Digital Twin. **PyArrow** (via Flight) handles the heavy lifting of streaming waveform data, preventing the control plane from being flooded by sensor noise.

### 3. In Practice (The How)

`coreason-signal` is designed to run close to the metal. Here is how it brings intelligence to the instrument loop.

#### The Reflex Engine: Reasoning on Logs
Instead of hard-coded error handlers, the `ReflexEngine` uses semantic similarity to find the right SOP for a given error context. It employs a "Dead Man's Switch" pattern to ensure that if the AI hangs, the system fails safely.

```python
from coreason_signal.edge_agent.reflex_engine import ReflexEngine
from coreason_signal.schemas import LogEvent

# The engine is initialized with a local vector store containing embedded SOPs
engine = ReflexEngine(vector_store=local_store, decision_timeout=0.2)

# An instrument throws a cryptic error
event = LogEvent(
    id="evt_123",
    timestamp="2023-10-27T10:00:00Z",
    level="ERROR",
    source="LiquidHandler-01",
    message="Aspiration timeout: pressure sensor delta < threshold",
    raw_code="ERR_VAC_04"
)

# The engine queries the SOPs and returns an actionable reflex
reflex = engine.decide(event)

if reflex and reflex.action == "RETRY":
    # The agent autonomously attempts recovery based on SOP-104
    print(f"Recovering: {reflex.reasoning}")
    # > Recovering: SOP-104 prescribes retry at 50% speed for vacuum errors.
```

#### The Soft Sensor: Virtual Probes
The `SoftSensorEngine` allows you to treat a machine learning model as just another hardware sensor. It enforces physics constraints (like min/max values) to ensure the AI's predictions remain within the bounds of reality.

```python
from coreason_signal.soft_sensor.engine import SoftSensorEngine
from coreason_signal.schemas import SoftSensorModel

# Define a virtual sensor for Cell Viability
model_config = SoftSensorModel(
    id="viability_pred_v1",
    input_sensors=["ph", "dissolved_oxygen", "rpm"],
    target_variable="viability_percent",
    physics_constraints={"min": 0.0, "max": 100.0},
    model_artifact=load_onnx_bytes("viability_pinn.onnx")
)

# Initialize the engine (automatically selects CPU/CUDA)
sensor = SoftSensorEngine(model_config)

# Run inference on live data stream
live_data = {"ph": 7.1, "dissolved_oxygen": 45.2, "rpm": 120.0}
result = sensor.infer(live_data)

print(f"Current Viability: {result['viability_percent']:.2f}%")
```

#### The Application Gateway
The main application orchestrates these components, exposing them via SiLA 2 for control and Arrow Flight for data streaming. It acts as the bridge between the physical instrument and the digital graph.

```python
from coreason_signal.main import Application

# The Application boots up the SiLA Gateway, Reflex Engine, and Vector Store
app = Application()
app.setup()

# It exposes the hardware capabilities (e.g., "Transfer", "Measure")
# while simultaneously running the "Nervous System" in the background
# to catch errors and stream insights.
app.run()
```
