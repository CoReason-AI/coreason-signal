# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

"""Configuration management for Coreason Signal."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the Coreason Signal application.

    Reads configuration from environment variables prefixed with 'SIGNAL_'.
    Also supports reading from a '.env' file.

    Attributes:
        LOG_LEVEL (str): Logging level (default: "INFO").
        SILA_PORT (int): Port for the SiLA 2 Server (default: 50052).
        ARROW_FLIGHT_PORT (int): Port for the Arrow Flight Server (default: 50055).
        REFLEX_TIMEOUT (float): Timeout in seconds for the Reflex Engine decision (default: 0.2).
        EMBEDDING_MODEL (str): Name of the embedding model for Vector Store (default: "BAAI/bge-small-en-v1.5").
        VECTOR_STORE_PATH (str): Path to the LanceDB vector store (default: "memory://").
        ONNX_PROVIDERS (list[str]): List of ONNX execution providers in priority order.
    """

    model_config = SettingsConfigDict(env_prefix="SIGNAL_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Logging
    LOG_LEVEL: str = "INFO"

    # SiLA / Connectivity
    SILA_PORT: int = 50052
    ARROW_FLIGHT_PORT: int = 50055

    # Edge Agent / Reflex Engine
    REFLEX_TIMEOUT: float = 0.2  # Seconds
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_STORE_PATH: str = "memory://"

    # Soft Sensor / ONNX
    ONNX_PROVIDERS: list[str] = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]


settings = Settings()
