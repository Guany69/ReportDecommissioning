from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.requests import CreateAnalysisRunRequest
from app.models.responses import CreateAnalysisRunResponse


router = APIRouter(
    prefix="/analysis-runs",
    tags=["Analysis Runs"],
)


@router.post(
    "",
    response_model=CreateAnalysisRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_analysis_run(
    request: CreateAnalysisRunRequest,
) -> CreateAnalysisRunResponse:
    del request

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Run persistence and queue processing are not configured yet."
        ),
    )
