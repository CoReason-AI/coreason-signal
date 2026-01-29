from unittest.mock import MagicMock, patch

import pytest
from coreason_signal.schemas import SoftSensorModel
from coreason_signal.soft_sensor.engine import SoftSensorEngine


@pytest.fixture
def mock_model_config() -> SoftSensorModel:
    return SoftSensorModel(
        id="test-sensor-complex",
        input_sensors=["input1"],
        target_variable="output1",
        physics_constraints={},
        model_artifact=b"fake-onnx-bytes",
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_empty_availability(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """
    Edge Case: get_available_providers returns empty list.
    Should fallback to ['CPUExecutionProvider'].
    """
    mock_get_providers.return_value = []

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=["CPUExecutionProvider"],
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_filters_unknown_providers(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """
    Complex Scenario: get_available_providers returns supported AND unsupported providers.
    Should strictly filter to only CUDA/OpenVINO + CPU, ignoring others (e.g. TensorRT).
    """
    mock_get_providers.return_value = [
        "TensorrtExecutionProvider",  # Unsupported/Ignored
        "CUDAExecutionProvider",  # Supported
        "MyCustomAccelerator",  # Ignored
        "CPUExecutionProvider",
    ]

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    # Expect TensorRT and Custom to be dropped
    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_initialization_failure_logging(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """
    Edge Case: InferenceSession raises an exception during init (e.g. invalid model).
    Should catch exception, log error, and raise RuntimeError.
    """
    mock_get_providers.return_value = ["CPUExecutionProvider"]

    # Simulate crash
    mock_session_cls.side_effect = Exception("Invalid ONNX format")

    with pytest.raises(RuntimeError, match="Failed to initialize inference session"):
        _ = SoftSensorEngine(mock_model_config)
