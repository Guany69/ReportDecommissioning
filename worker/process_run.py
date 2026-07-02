"""Worker entry point: pull run jobs from the queue and process them.

Run as a separate process from the API:  python -m worker.process_run

Each job resolves its input files (OneDrive), invokes the analysis service,
persists results, and updates run status. The processing loop is implemented in
the async processing phase; this module defines its shape.
"""
from __future__ import annotations

from app.services.analysis_service import AnalysisService
from app.services.onedrive_service import OneDriveService
from app.services.queue_service import QueueService
from app.services.result_service import ResultService
from app.settings import get_settings


def process_once(job: dict) -> None:
    """Process a single dequeued run job. Implemented in the processing phase."""
    raise NotImplementedError("process_once is implemented in the async processing phase.")


def main() -> None:
    """Continuously consume and process run jobs from the queue."""
    settings = get_settings()
    _ = (
        QueueService(settings),
        OneDriveService(settings),
        AnalysisService(settings),
        ResultService(settings),
    )
    raise NotImplementedError("The worker loop is implemented in the async processing phase.")


if __name__ == "__main__":
    main()
