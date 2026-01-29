# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from typing import Dict

import numpy as np
import onnxruntime as ort

from coreason_signal.config import settings
from coreason_signal.schemas import SoftSensorModel
from coreason_signal.utils.logger import logger


class SoftSensorEngine:
    """Engine for executing Soft Sensor models (PINNs) using ONNX Runtime.

    Handles inference and physics constraint enforcement. It dynamically selects
    the best available hardware accelerator (CUDA, OpenVINO, CPU).

    Attributes:
        config (SoftSensorModel): The model configuration.
    """

    def __init__(self, model_config: SoftSensorModel):
        """Initialize the Soft Sensor Engine.

        Args:
            model_config (SoftSensorModel): The SoftSensorModel configuration containing
                the model artifact and metadata.
        """
        self.config = model_config
        self._session = self._load_session()
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._constraints = self._parse_constraints()

    def _load_session(self) -> ort.InferenceSession:
        """Load the ONNX inference session from the model artifact bytes.

        Detects available hardware acceleration and prioritizes them based on configuration.

        Returns:
            ort.InferenceSession: The initialized ONNX Runtime session.

        Raises:
            RuntimeError: If session initialization fails.
        """
        try:
            available = set(ort.get_available_providers())
            logger.info(f"Available ONNX providers: {available}")

            # Intersect config preference with availability, preserving config order
            selected_providers = [p for p in settings.ONNX_PROVIDERS if p in available]

            # Always ensure CPU fallback at the end
            if "CPUExecutionProvider" not in selected_providers:
                selected_providers.append("CPUExecutionProvider")

            logger.info(f"Selected ONNX providers for {self.config.id}: {selected_providers}")

            return ort.InferenceSession(self.config.model_artifact, providers=selected_providers)
        except Exception as e:
            logger.error(f"Failed to load ONNX model for sensor {self.config.id}: {e}")
            raise RuntimeError(f"Failed to initialize inference session: {e}") from e

    def _parse_constraints(self) -> Dict[str, float]:
        """Parse physics constraints into usable float values.

        Supports keys starting with 'min' (lower bound) and 'max' (upper bound).

        Returns:
            Dict[str, float]: Dictionary of parsed constraints (keys: 'min', 'max').

        Raises:
            ValueError: If min constraint is greater than max constraint.
        """
        parsed: Dict[str, float] = {}
        for key, value in self.config.physics_constraints.items():
            if key.lower().startswith("min"):
                parsed["min"] = value
            elif key.lower().startswith("max"):
                parsed["max"] = value

        if "min" in parsed and "max" in parsed and parsed["min"] > parsed["max"]:
            raise ValueError(f"Invalid constraints: min ({parsed['min']}) > max ({parsed['max']})")

        return parsed

    def infer(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Run inference on the provided inputs.

        Args:
            inputs (Dict[str, float]): Dictionary mapping sensor names to values.

        Returns:
            Dict[str, float]: Dictionary containing the target variable and its predicted value.

        Raises:
            ValueError: If required input sensors are missing.
            RuntimeError: If inference execution fails.
        """
        # 1. Validate inputs
        missing = [s for s in self.config.input_sensors if s not in inputs]
        if missing:
            raise ValueError(f"Missing required input sensors: {missing}")

        # 2. Prepare input vector
        # We assume the model expects a single input tensor of shape (1, n_features)
        # corresponding to the order in config.input_sensors.
        feature_vector = [inputs[s] for s in self.config.input_sensors]
        input_tensor = np.array([feature_vector], dtype=np.float32)

        # 3. Run Inference
        try:
            outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
            # robustly extract scalar value from tensor
            raw_prediction = float(outputs[0].item())
        except Exception as e:
            logger.error(f"Inference failed for {self.config.id}: {e}")
            raise RuntimeError(f"Inference execution failed: {e}") from e

        # 4. Apply Physics Constraints
        final_prediction = self._apply_constraints(raw_prediction)

        return {self.config.target_variable: final_prediction}

    def _apply_constraints(self, value: float) -> float:
        """Apply min/max physics constraints to the prediction.

        Args:
            value (float): The raw predicted value.

        Returns:
            float: The constrained value.
        """
        if "min" in self._constraints and value < self._constraints["min"]:
            logger.info(f"Constraint active: Clipped {value} to min {self._constraints['min']}")
            return self._constraints["min"]
        if "max" in self._constraints and value > self._constraints["max"]:
            logger.info(f"Constraint active: Clipped {value} to max {self._constraints['max']}")
            return self._constraints["max"]
        return value

    def update_constraints(self, new_constraints: Dict[str, float]) -> None:
        """Update the physics constraints at runtime.

        Args:
            new_constraints (Dict[str, float]): New constraints to apply.
        """
        logger.info(f"Updating physics constraints for {self.config.id}: {new_constraints}")
        self.config.physics_constraints = new_constraints
        self._constraints = self._parse_constraints()
