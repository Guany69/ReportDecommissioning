from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.responses import HealthResponse
from app.settings import get_settings


router = APIRouter(tags=["Health"])


@router.get(
    "/health/live",
    response_model=HealthResponse,
)
async def liveness() -> HealthResponse:
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
)
async def readiness(
    request: Request,
) -> HealthResponse:
    settings = get_settings()

    engine_config = getattr(
        request.app.state,
        "engine_config",
        None,
    )

    if engine_config is None:
        raise HTTPException(
            status_code=503,
            detail="Engine configuration is not loaded.",
        )

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
    )
