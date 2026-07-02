"""Enqueue and dequeue analysis-run jobs for asynchronous processing.

The API enqueues a job when a run is submitted; the worker (`worker.process_run`)
consumes it. Backed by a durable queue (e.g. Azure Storage Queue) configured in
Settings. Implemented in the async processing phase.
"""
from __future__ import annotations

from app.settings import Settings


class QueueService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def enqueue(self, run_id: str, payload: dict) -> None:
        """Publish a run job for the worker to pick up."""
        raise NotImplementedError("QueueService.enqueue is implemented in the processing phase.")

    def dequeue(self):
        """Yield the next available run job (worker side)."""
        raise NotImplementedError("QueueService.dequeue is implemented in the processing phase.")
