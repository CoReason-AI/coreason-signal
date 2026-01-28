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

import argparse
import json
import signal
import sys
from types import FrameType
from typing import Optional

from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr
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
    parser = argparse.ArgumentParser(description="Coreason Signal CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a signal")
    ingest_parser.add_argument("data", help="JSON string of the signal data")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query signals")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of results")

    # Serve command
    subparsers.add_parser("serve", help="Start the service")

    args = parser.parse_args()

    # Establish Identity Context
    system_context = UserContext(
        user_id=SecretStr("cli-user"),
        roles=["system"],
        metadata={"source": "cli"}
    )

    svc = Service()

    # Register signal handlers
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        with svc:
            if args.command == "ingest":
                try:
                    data = json.loads(args.data)
                    svc.ingest_signal(data, system_context)
                    print("Signal ingested successfully.")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON data provided.")
                    sys.exit(1)

            elif args.command == "query":
                results = svc.query_signals(args.query, args.top_k, system_context)
                # Serialize results (assuming they are Pydantic models or dicts)
                output = []
                for res in results:
                    if hasattr(res, "model_dump"):
                        output.append(res.model_dump())
                    else:
                        output.append(res)
                print(json.dumps(output, indent=2, default=str))

            elif args.command == "serve":
                logger.info(
                    "Starting service with system context...",
                    user_id=system_context.user_id.get_secret_value(),
                )
                svc.run_forever(context=system_context)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main.")
    except Exception as e:
        logger.exception(f"Fatal application error: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
