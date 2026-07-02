"""Analysis-run endpoints.

A "run" is one execution of the decommissioning engine over a set of Workday
exports. Submitting a run is asynchronous: the API validates the request,
enqueues the job, and returns a run id the caller polls for status/results.

The handlers below are scaffolding for the migration — the request/response
contracts are defined, and the service wiring is filled in during the async
processing phase. Until then they return HTTP 501 rather than pretending to work.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.requests import CreateRunRequest
from app.models.responses import RunResponse, RunResultResponse

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
def create_run(request: CreateRunRequest) -> RunResponse:
    """Submit a new analysis run (enqueued for the worker to process)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Run submission is not implemented yet (async processing phase).",
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    """Return the status of a previously submitted run."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Run status lookup is not implemented yet.",
    )


@router.get("/{run_id}/result", response_model=RunResultResponse)
def get_run_result(run_id: str) -> RunResultResponse:
    """Return the finished result payload for a completed run."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Run result retrieval is not implemented yet.",
    )
