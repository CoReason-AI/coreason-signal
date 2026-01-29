from typing import Generator, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from coreason_signal.schemas import SoftSensorModel
from coreason_signal.soft_sensor.engine import SoftSensorEngine


# Sample config for testing
@pytest.fixture
def sample_model_config() -> SoftSensorModel:
    return SoftSensorModel(
        id="test_model",
        input_sensors=["temp", "ph"],
        target_variable="growth_rate",
        physics_constraints={"min_growth": 0.0, "max_growth": 10.0},
        model_artifact=b"fake_onnx_bytes",
    )


@pytest.fixture
def mock_ort_session() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    with patch("coreason_signal.soft_sensor.engine.ort.InferenceSession") as mock_cls:
        session_instance = MagicMock()
        # Mock input/output metadata
        input_meta = MagicMock()
        input_meta.name = "input_0"
        session_instance.get_inputs.return_value = [input_meta]

        output_meta = MagicMock()
        output_meta.name = "output_0"
        session_instance.get_outputs.return_value = [output_meta]

        mock_cls.return_value = session_instance
        yield mock_cls, session_instance


def test_init(sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    mock_cls, session_instance = mock_ort_session

    engine = SoftSensorEngine(sample_model_config)

    mock_cls.assert_called_once_with(b"fake_onnx_bytes", providers=["CPUExecutionProvider"])
    assert engine._input_name == "input_0"
    assert engine._output_name == "output_0"
    assert engine._constraints == {"min": 0.0, "max": 10.0}


def test_infer_success(sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    # Mock inference output: a numpy array containing [[5.0]]
    session_instance.run.return_value = [np.array([[5.0]], dtype=np.float32)]

    inputs = {"temp": 37.0, "ph": 7.0}
    result = engine.infer(inputs)

    assert result == {"growth_rate": 5.0}

    # Verify input tensor construction
    args, _ = session_instance.run.call_args
    output_names, input_feed = args
    assert output_names == ["output_0"]
    assert "input_0" in input_feed

    input_tensor = input_feed["input_0"]
    assert input_tensor.shape == (1, 2)
    assert np.allclose(input_tensor, np.array([[37.0, 7.0]], dtype=np.float32))


def test_infer_missing_input(
    sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]
) -> None:
    _, _ = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    inputs = {"temp": 37.0}  # Missing ph

    with pytest.raises(ValueError, match="Missing required input sensors"):
        engine.infer(inputs)


def test_infer_constraints_min(
    sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]
) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    # Output -1.0, should be clipped to 0.0
    session_instance.run.return_value = [np.array([[-1.0]], dtype=np.float32)]

    inputs = {"temp": 37.0, "ph": 7.0}
    result = engine.infer(inputs)

    assert result["growth_rate"] == 0.0


def test_infer_constraints_max(
    sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]
) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    # Output 15.0, should be clipped to 10.0
    session_instance.run.return_value = [np.array([[15.0]], dtype=np.float32)]

    inputs = {"temp": 37.0, "ph": 7.0}
    result = engine.infer(inputs)

    assert result["growth_rate"] == 10.0


def test_infer_runtime_error(
    sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]
) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    session_instance.run.side_effect = Exception("ONNX Error")

    inputs = {"temp": 37.0, "ph": 7.0}

    with pytest.raises(RuntimeError, match="Inference execution failed"):
        engine.infer(inputs)


def test_init_failure(sample_model_config: SoftSensorModel) -> None:
    with patch("coreason_signal.soft_sensor.engine.ort.InferenceSession") as mock_cls:
        mock_cls.side_effect = Exception("Invalid Model")

        with pytest.raises(RuntimeError, match="Failed to initialize inference session"):
            SoftSensorEngine(sample_model_config)


def test_invalid_constraint_parsing(mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    # Test with config that has non-min/max or weird keys
    # Schema validation ensures values are numbers, but keys are open strings.

    config = SoftSensorModel(
        id="test_model_2",
        input_sensors=["a"],
        target_variable="y",
        physics_constraints={"other_param": 100.0, "some_key": 50.0},
        model_artifact=b"bytes",
    )

    engine = SoftSensorEngine(config)
    assert engine._constraints == {}


def test_conflicting_constraints(mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    config = SoftSensorModel(
        id="test_model_bad",
        input_sensors=["a"],
        target_variable="y",
        physics_constraints={"min_y": 10.0, "max_y": 5.0},  # Min > Max
        model_artifact=b"bytes",
    )
    with pytest.raises(ValueError, match="Invalid constraints: min"):
        SoftSensorEngine(config)


def test_infer_extra_inputs(
    sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]
) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)
    session_instance.run.return_value = [np.array([[5.0]], dtype=np.float32)]

    inputs = {"temp": 37.0, "ph": 7.0, "extra": 99.9}
    result = engine.infer(inputs)
    assert result == {"growth_rate": 5.0}


def test_infer_nan_output(sample_model_config: SoftSensorModel, mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    _, session_instance = mock_ort_session
    engine = SoftSensorEngine(sample_model_config)

    # Model returns NaN
    session_instance.run.return_value = [np.array([[np.nan]], dtype=np.float32)]

    inputs = {"temp": 37.0, "ph": 7.0}
    result = engine.infer(inputs)

    # NaN should pass through as constraints checks (NaN < min) are False
    assert np.isnan(result["growth_rate"])


def test_infer_zero_inputs(mock_ort_session: Tuple[MagicMock, MagicMock]) -> None:
    _, session_instance = mock_ort_session
    config = SoftSensorModel(
        id="bias_model",
        input_sensors=[],
        target_variable="y",
        physics_constraints={},
        model_artifact=b"bytes",
    )
    engine = SoftSensorEngine(config)
    session_instance.run.return_value = [np.array([[1.0]], dtype=np.float32)]

    inputs: dict[str, float] = {}
    result = engine.infer(inputs)
    assert result == {"y": 1.0}
