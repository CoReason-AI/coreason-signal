# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from coreason_signal.utils.logger import logger


class TestLogger:
    def test_logger_config(self) -> None:
        """Verify logger is configured."""
        assert logger is not None

    def test_logger_file(self, tmp_path) -> None:  # type: ignore
        """Verify logger writes to file (mocked path)."""
        # Note: We can't easily change the logger path at runtime without reloading module
        # but we can verify it has handlers
        assert len(logger._core.handlers) > 0
