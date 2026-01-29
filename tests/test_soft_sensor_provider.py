from unittest.mock import MagicMock, patch

import pytest

from coreason_signal.schemas import SoftSensorModel
from coreason_signal.soft_sensor.engine import SoftSensorEngine


@pytest.fixture
def mock_model_config() -> SoftSensorModel:
    return SoftSensorModel(
        id="test-sensor",
        input_sensors=["input1"],
        target_variable="output1",
        physics_constraints={},
        model_artifact=b"fake-onnx-bytes",
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_cuda(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """Test that CUDA is prioritized when available."""
    mock_get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    # Verify InferenceSession called with CUDA first
    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_openvino(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """Test that OpenVINO is prioritized when available (and no CUDA)."""
    mock_get_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    # Verify InferenceSession called with OpenVINO first
    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_cpu_only(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """Test fallback to CPU when no accelerators are available."""
    mock_get_providers.return_value = ["CPUExecutionProvider"]

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    # Verify InferenceSession called with only CPU
    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=["CPUExecutionProvider"],
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_provider_selection_cuda_and_openvino(
    mock_get_providers: MagicMock,
    mock_session_cls: MagicMock,
    mock_model_config: SoftSensorModel,
) -> None:
    """Test prioritization when multiple accelerators are available."""
    mock_get_providers.return_value = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Setup mock session
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input1")]
    mock_session.get_outputs.return_value = [MagicMock(name="output1")]
    mock_session_cls.return_value = mock_session

    _ = SoftSensorEngine(mock_model_config)

    # Verify InferenceSession called with both, CUDA first
    mock_session_cls.assert_called_with(
        mock_model_config.model_artifact,
        providers=[
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
