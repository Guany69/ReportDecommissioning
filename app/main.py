from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import health, runs
from app.settings import get_settings
from report_cleanup.config import load_config


settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s %(message)s"
    ),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Report Decommissioning API")

    # Fail during startup if config.yaml is missing or invalid.
    app.state.engine_config = load_config(
        settings.engine_config_path
    )

    logger.info("Engine configuration loaded")

    yield

    logger.info("Stopping Report Decommissioning API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url=(
        "/openapi.json"
        if settings.enable_docs
        else None
    ),
)


app.include_router(health.router)
app.include_router(runs.router, prefix="/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request,
    exc: Exception,
):
    logger.exception(
        "Unhandled API error",
        extra={"path": request.url.path},
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "The request could not be completed.",
        },
    )
