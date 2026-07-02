"""Read access to a completed run's analysis results.

Wraps the engine's SQLite tables (reports, report_reasons, dup_groups, …) so the
result service can shape API/Power BI payloads. Implemented in a later phase.
"""
from __future__ import annotations


class ResultRepository:
    def get_records(self, run_id: str) -> list[dict]:
        """Return per-report result rows for a completed run."""
        raise NotImplementedError

    def get_summary(self, run_id: str) -> dict:
        """Return aggregate counts (totals, hard-rule, duplicate groups)."""
        raise NotImplementedError
