"""Assemble result payloads from persisted run artifacts.

Reads a completed run's records/groups (via the repositories) and shapes them
into the API response models and Power BI-facing outputs. Implemented in the
Azure SQL / results phase.
"""
from __future__ import annotations

from app.settings import Settings


class ResultService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_result(self, run_id: str) -> dict:
        """Return the assembled result envelope for a completed run."""
        raise NotImplementedError("ResultService.get_result is implemented in a later phase.")
