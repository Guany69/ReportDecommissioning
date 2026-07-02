"""Persistence for run lifecycle records (id, status, timestamps).

Tracks a run from `queued` through `completed`/`failed` so the API can answer
status queries. Implemented in the async processing phase.
"""
from __future__ import annotations

from app.models.responses import RunStatus


class RunRepository:
    def create(self, run_id: str, payload: dict) -> None:
        """Persist a new run in the `queued` state."""
        raise NotImplementedError

    def set_status(self, run_id: str, status: RunStatus, detail: str | None = None) -> None:
        """Update a run's lifecycle status."""
        raise NotImplementedError

    def get(self, run_id: str) -> dict | None:
        """Return the stored run record, or None if unknown."""
        raise NotImplementedError
