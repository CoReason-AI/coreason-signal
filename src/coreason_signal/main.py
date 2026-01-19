# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

"""Main application entry point for Coreason Signal."""

import signal
from types import FrameType
from typing import Optional

from coreason_signal.service import Service
from coreason_signal.utils.logger import logger


def _shutdown_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Signal handler to trigger graceful shutdown.

    Raises KeyboardInterrupt which is caught by the main loop.
    """
    logger.info(f"Signal {signum} received. Stopping services...")
    raise KeyboardInterrupt


def main() -> None:
    """Entry point for the application.

    Sets up signal handlers and runs the Application via the Service facade.
    """
    svc = Service()

    # Register signal handlers
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        with svc:
            svc.run_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main.")
    except Exception as e:
        logger.exception(f"Fatal application error: {e}")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
