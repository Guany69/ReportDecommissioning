from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


RunStatus = Literal[
    "submitted",
    "queued",
    "downloading",
    "validating",
    "processing",
    "persisting_results",
    "completed",
    "failed_validation",
    "failed_processing",
]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class CreateAnalysisRunResponse(BaseModel):
    run_id: UUID
    submission_id: str
    status: RunStatus
    submitted_at: datetime


class ErrorResponse(BaseModel):
    error: str
    message: str
    correlation_id: str | None = None
