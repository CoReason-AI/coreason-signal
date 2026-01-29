# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_signal

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request

from coreason_signal.schemas import AgentReflex
from coreason_signal.utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    from coreason_signal.service import ServiceAsync


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Retrieve the service instance from app state
    service = cast("ServiceAsync | None", getattr(app.state, "service", None))
    if not service:
        logger.error("Service instance not found in app.state during startup.")
        yield
        return

    logger.info("API Sidecar starting up. Initializing ServiceAsync...")
    # Setup and Start are handled here to ensure they run in the event loop managed by uvicorn
    await service.setup()
    await service.start()

    yield

    logger.info("API Sidecar shutting down. Stopping ServiceAsync...")
    await service.shutdown()


router = APIRouter()


@router.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    """Return the current state of the gateway."""
    service = cast("ServiceAsync | None", getattr(request.app.state, "service", None))
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not service.gateway or not service.gateway.device_def:
        raise HTTPException(status_code=503, detail="Gateway not ready")

    return {
        "device_id": service.gateway.device_def.id,
        "sila_port": service.gateway.port,
        # flight_server.location is typically a string "grpc://..."
        "arrow_flight_port": service.flight_server.location if service.flight_server else None,
        "status": "active",
    }


@router.get("/sensors/latest")
async def get_latest_sensors(request: Request) -> Dict[str, Any]:
    """Return a summary of the buffered record batches."""
    service = cast("ServiceAsync | None", getattr(request.app.state, "service", None))
    if not service or not service.flight_server:
        raise HTTPException(status_code=503, detail="Flight server not available")

    try:
        batches = service.flight_server.get_latest_data()
        summary = []
        for batch in batches:
            summary.append(
                {
                    "num_rows": batch.num_rows,
                    "num_columns": batch.num_columns,
                    "schema": str(batch.schema),
                    "nbytes": batch.nbytes,
                }
            )
        return {"buffered_batches_count": len(batches), "batches": summary}
    except Exception as e:
        logger.error(f"Failed to get latest sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reflex/trigger")
async def trigger_reflex(reflex: AgentReflex, request: Request) -> Dict[str, Any]:
    """Manually trigger a reflex action."""
    service = cast("ServiceAsync | None", getattr(request.app.state, "service", None))
    if not service or not service.reflex_engine:
        raise HTTPException(status_code=503, detail="Reflex engine not available")

    service.reflex_engine.trigger(reflex)
    return {"status": "triggered", "reflex": reflex}


@router.put("/soft-sensor/constraints")
async def update_constraints(constraints: Dict[str, float], request: Request) -> Dict[str, Any]:
    """Update the SoftSensorEngine configuration at runtime."""
    service = cast("ServiceAsync | None", getattr(request.app.state, "service", None))
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not service.soft_sensor_engine:
        # If the engine is not initialized (e.g. no model), we cannot update it.
        raise HTTPException(status_code=503, detail="Soft Sensor Engine not active")

    try:
        service.soft_sensor_engine.update_constraints(constraints)
        return {"status": "updated", "constraints": constraints}
    except Exception as e:
        logger.error(f"Failed to update constraints: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e


app: FastAPI = FastAPI(
    title="Coreason Signal Management API",
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(router)
