"""Outbound response models."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class RunStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RunResponse(BaseModel):
    """Status envelope for a submitted run."""

    run_id: str
    status: RunStatus
    detail: str | None = None


class RunResultResponse(BaseModel):
    """Result envelope for a completed run.

    Holds summary counts plus references to the generated artifacts; the full
    per-report table is fetched separately (or read by Power BI from storage).
    """

    run_id: str
    status: RunStatus
    total_reports: int = 0
    hard_rule_count: int = 0
    duplicate_group_count: int = 0
    workbook_ref: str | None = None
    summary_workbook_ref: str | None = None
    database_ref: str | None = None
