# Product Requirements Document: coreason-signal

**Domain:** Edge AI, Self-Driving Labs (SDL), Soft Sensing, & Lab Proprioception
**Architectural Role:** The "Nervous System" / The Edge Cortex
**Core Philosophy:** "The Lab is a Stream, not a File. Reason at the Edge. Close the Loop Locally."
**Dependencies:** coreason-graph-nexus (Digital Twin), coreason-chronos (Forecasting), sila2-lib (Connectivity), onnxruntime (Edge Inference)

---

## 1. Executive Summary

**coreason-signal** is the high-throughput **Edge Intelligence Gateway** for the CoReason ecosystem. It bridges the "Air Gap" between the cloud-based reasoning of coreason-cortex and the physical reality of laboratory instruments.

It transforms the platform from a "Passive Logger" into an **"Active Laboratory Participant."** By deploying **Micro-LLMs** and **Physics-Informed Neural Networks (PINNs)** directly to the edge hardware, signal enables instruments to understand their own errors, infer invisible biological states (Soft Sensing), and execute sub-second recovery protocols without waiting for cloud intervention. It is the engine that powers the **Self-Driving Lab**.

---

## 2. Functional Philosophy

The agent must implement the **Sense-Infer-Act-Sync Loop**:

1.  **Edge Agentic AI (SOTA):** We move beyond simple "Threshold Alerts." We deploy **Micro-LLMs** (Small Language Models) to the instrument controller. These "Nano-Agents" parse unstructured error logs, query local SOPs using RAG, and autonomously trigger recovery actions (e.g., "Unclog Pipette Tip") for transient faults.
2.  **Soft Sensing (Virtual Metrology):** Biology is often invisible to hardware sensors. We use **Soft Sensors**—AI models that infer Critical Quality Attributes (like Cell Viability) from secondary physical signals (pH, Dissolved Oxygen, Agitation). This turns a standard bioreactor into a "Smart" device.
3.  **Protocol Polyglot:** The system natively adopts **SiLA 2** (Standard in Lab Automation) as the universal schema. It acts as a bridge, wrapping legacy protocols (OPC-UA, Serial, Modbus) and even analog gauges (via Computer Vision) into clean SiLA 2 microservices.
4.  **Live Digital Twin:** It syncs the physical state to coreason-graph-nexus using **Delta Updates**. It ensures the Knowledge Graph reflects the *current* reality (latency < 1s), not yesterday's report.

---

## 3. Core Functional Requirements (Component Level)

### 3.1 The Edge Agent Host (The Micro-Cortex)

**Concept:** A local container for running quantized Micro-LLMs (e.g., Phi-4, Llama-Edge).

*   **Role:** The "Reflex Arc." Handles immediate tactical decisions.
*   **Capabilities:**
    *   **Semantic Log Parsing:** Reads vendor-specific error codes (Err: 0x4F) and translates them into semantic meaning ("Tip Mismatch").
    *   **Local RAG:** Queries a lightweight, in-memory vector store containing specific "Emergency SOPs" to determine if a shutdown is required.
    *   **Autonomy Level:** Configurable. Can be set to "Notify Only" (Human Loop) or "Autonomous Recovery" (Self-Healing).

### 3.2 The Soft Sensor Engine (The Virtual Probe)

**Concept:** Infers unmeasurable biological states in real-time.

*   **Mechanism:** Runs **Physics-Informed Neural Networks (PINNs)** via ONNX Runtime.
*   **Input:** Streaming vectors of `[T, pH, DO_2, RPM]`.
*   **Logic:** Uses metabolic kinetic models constrained by physics to predict outputs.
*   **Output:** Virtual Streams: `Predicted_Glucose_Conc`, `Predicted_Viability`.
*   **Value:** Allows the system to make decisions based on *Biology* (Growth Phase), not just *Physics* (Temperature).

### 3.3 The Protocol Bridge (The Translator)

**Concept:** Universal Hardware Abstraction Layer (HAL).

*   **SiLA 2 Native:** Uses Feature Discovery to auto-map instrument capabilities.
*   **Legacy Wrappers:**
    *   **Serial-to-SiLA:** Wraps RS-232 scales as SiLA 2 services.
    *   **Vision-to-SiLA:** Uses a webcam and lightweight OCR/Computer Vision to read analog gauges and publish the value as a digital stream.
*   **Streaming:** Uses **Apache Arrow Flight** to buffer high-frequency waveform data (e.g., 1000Hz vibration logs) locally, pushing only aggregates to the cloud to prevent bandwidth saturation.

### 3.4 The Twin Syncer (The Reality Link)

**Concept:** Updates the Knowledge Graph.

*   **Logic:**
    *   **Delta Throttling:** Only writes to `coreason-graph-nexus` when a value changes beyond a significant sigma threshold (noise filtering).
    *   **Fact Promotion:** Promotes raw data into Semantic Facts.
        *   *Raw:* pH = 6.2.
        *   *Fact:* `(:Bioreactor)-[:STATE_CHANGE]->(Acidic_Stress)`.

---

## 4. Integration Requirements

*   **coreason-graph-nexus:**
    *   Stores the **Digital Twin**. signal is the authoritative writer for "Current State" properties on Instrument Nodes.
*   **coreason-connect:**
    *   signal triggers *Reflexes* (Stop Motor), but calls connect for *Transactions* (Order Reagent, Log Service Ticket).
*   **coreason-chronos:**
    *   signal feeds Soft Sensor outputs to chronos.
    *   *Flow:* Sensors -> Signal (Current Viability) -> Chronos (Forecasted Viability in 48h) -> Cortex (Harvest Decision).

---

## 5. User Stories

### Story A: The "Self-Healing" Robot (Edge Agent)

**Context:** A liquid handler fails to aspirate because of a micro-clog.
**Error:** Instrument throws `ERR_VACUUM_PRESSURE_LOW`.
**Edge Agent:** Intercepts error. Queries Local SOP vector store.
**Reasoning:** "SOP-104 states: For vacuum errors, retry aspiration once at 50% speed."
**Action:** Sends command `Aspirate(speed=0.5)`.
**Result:** The clot clears. The run is saved. Cortex is notified of the "Transient Fault" for post-run analysis.

### Story B: The "Virtual Assay" (Soft Sensor)

**Context:** Scientist needs to stop the bioreactor exactly when the drug titer peaks.
**Problem:** Titer measurement requires a 24-hour external HPLC assay.
**Soft Sensor:** Analyzes real-time "Off-Gas" (CO2 evolution) and pH consumption rates.
**Inference:** "Predicted Titer: 1.2 g/L (Peak Reached)."
**Action:** signal triggers the "Harvest" notification immediately.

### Story C: The "Vision" Legacy Hook

**Context:** An ancient incubator has no data port, only a digital display.
**Signal Action:** Connects to a $20 webcam pointed at the screen.
**Edge Vision:** Runs a quantized OCR model (10ms inference).
**Output:** Publishes Temperature = 37.0°C via SiLA 2.
**Result:** Digital Twin visibility for legacy hardware.

---

## 6. Data Schema

### DeviceDefinition (SiLA 2 Mapping)

```python
class DeviceDefinition(BaseModel):
    id: str                 # "LiquidHandler-01"
    driver_type: str        # "SiLA2", "SerialWrapper", "VisionWrapper"
    endpoint: str           # "https://192.168.1.50:50052"
    capabilities: List[str] # ["Transfer", "Wash", "Heater"]

    # Edge AI Config
    edge_agent_model: str   # "phi-4-quantized.onnx"
    allowed_reflexes: List[str] # ["RETRY", "PAUSE", "ABORT"]
```

### SoftSensorModel

```python
class SoftSensorModel(BaseModel):
    id: str                 # "model_titer_pred_v2"
    input_sensors: List[str] # ["ph", "do2", "agitation"]
    target_variable: str    # "titer_g_L"
    physics_constraints: Dict[str, str] # {"min_titer": "0.0"}
    model_artifact: bytes   # The ONNX file
```

---

## 7. Implementation Directives for the Coding Agent

1.  **Hardware Acceleration:**
    *   Detect host capabilities (NVIDIA Jetson, Coral TPU, or standard CPU).
    *   Use `onnxruntime-gpu` or `openvino` to accelerate Micro-LLM and Soft Sensor inference.
2.  **Streaming Architecture:**
    *   Use **Apache Arrow Flight** for moving high-frequency sensor data. Do *not* use HTTP/JSON for waveform streaming (vibration/voltammetry); it is too slow.
3.  **Local Vector Store:**
    *   Embed a lightweight vector database (e.g., **Chroma** or **LanceDB**) directly inside the package for the Edge Agent's RAG. It must run in-process, without an external server.
4.  **Safety Watchdog:**
    *   Implement a strict "Dead Man's Switch." If the Edge Agent hangs for >200ms during a decision, the system must default to "PAUSE" and alert the cloud.
