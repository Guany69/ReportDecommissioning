"""Bridge between the API/worker and the existing report_cleanup engine.

This is the single place that invokes `run_pipeline`, so the web layer never
imports engine internals directly. Implemented in the async processing phase.
"""
from __future__ import annotations

from app.settings import Settings


class AnalysisService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, table1_path: str, table2_path: str | None,
            table3_fields_path: str | None, out_dir: str) -> dict:
        """Execute the decommissioning engine over the given input files.

        Wraps report_cleanup.pipeline.run_pipeline. Wired up in a later phase.
        """
        raise NotImplementedError("AnalysisService.run is implemented in the processing phase.")
