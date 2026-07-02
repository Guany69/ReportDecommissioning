"""Liveness / readiness endpoint.

Kept dependency-free so load balancers and Azure App Service health probes can
hit it cheaply without touching the engine, queue, or storage.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
