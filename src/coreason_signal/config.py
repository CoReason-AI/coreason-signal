# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[misc]
    """
    Centralized configuration for the Coreason Signal application.
    Reads from environment variables (prefix 'SIGNAL_').
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
