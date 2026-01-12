# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from unittest.mock import patch

from coreason_signal.utils.logger import logger, setup_logger


def test_logger_setup() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""

    with patch("coreason_signal.utils.logger.logger") as mock_logger:
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                # Scenario 1: Directory does not exist
                mock_exists.return_value = False
                setup_logger()

                # Check remove called
                mock_logger.remove.assert_called()

                # Check add called (at least twice)
                assert mock_logger.add.call_count >= 2

                # Check directory creation
                mock_mkdir.assert_called_with(parents=True, exist_ok=True)


def test_logger_setup_dir_exists() -> None:
    """Test logger setup when directory already exists."""
    with patch("coreason_signal.utils.logger.logger"):
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                # Scenario 2: Directory exists
                mock_exists.return_value = True
                setup_logger()

                mock_mkdir.assert_not_called()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
    assert setup_logger is not None
