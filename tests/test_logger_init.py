# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

import importlib
import shutil
import sys
from pathlib import Path


def test_logger_creates_directory() -> None:
    """Test that importing the logger module creates the logs directory."""
    log_path = Path("logs")

    # 1. Ensure clean state
    # If logger is already initialized (by other tests), it holds a lock on the file.
    if "coreason_signal.utils.logger" in sys.modules:
        from coreason_signal.utils.logger import logger

        logger.remove()

    if log_path.exists():
        shutil.rmtree(log_path)

    assert not log_path.exists()

    # 2. Reload the module (force execution of top-level code)
    if "coreason_signal.utils.logger" in sys.modules:
        import coreason_signal.utils.logger

        importlib.reload(coreason_signal.utils.logger)
    else:
        import coreason_signal.utils.logger

    # 3. Verify directory creation
    assert log_path.exists()
    assert log_path.is_dir()
