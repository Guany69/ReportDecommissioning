"""Assemble result payloads from persisted run artifacts.

Reads a completed run's records/groups (via the repositories) and shapes them
into the API response models and Power BI-facing outputs. Implemented later.
"""
from __future__ import annotations

from app.models.responses import RunResultResponse
from app.settings import Settings


class ResultService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_result(self, run_id: str) -> RunResultResponse:
        """Return the assembled result envelope for a completed run."""
        raise NotImplementedError("ResultService.get_result is implemented in a later phase.")
