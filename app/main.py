"""FastAPI application entry point — import target `app.main:app`.

Run locally:  uvicorn app.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api import health, runs
from app.settings import get_settings


def create_app() -> FastAPI:
    """Application factory. Wires routers and shared configuration."""
    settings = get_settings()
    application = FastAPI(
        title="Report Decommissioning API",
        version="0.1.0",
        summary="API front end for the report decommissioning engine.",
    )
    application.state.settings = settings
    application.include_router(health.router)
    application.include_router(runs.router)
    return application


app = create_app()
