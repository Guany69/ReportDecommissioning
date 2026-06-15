"""Validation: warn where possible, abort only when the engine cannot proceed."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_coverage: dict[str, float] = field(default_factory=dict)

    def has_fatal(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARN:  {w}")
        return "\n".join(lines) if lines else "No validation issues."


def coverage(df: pd.DataFrame) -> dict[str, float]:
    """Percent populated (non-null, non-blank) per column."""
    out = {}
    n = max(len(df), 1)
    for col in df.columns:
        filled = df[col].map(lambda v: v is not None and str(v).strip() != "" and not pd.isna(v)).sum()
        out[col] = round(100.0 * filled / n, 1)
    return out


def validate(t1: pd.DataFrame, t2: pd.DataFrame | None, t1_map, t2_map) -> ValidationReport:
    vr = ValidationReport()

    # Table 1 must exist with rows and a report name column.
    if t1 is None or len(t1) == 0:
        vr.errors.append("Table 1 (metadata) has no data rows.")
        return vr
    if "report_name" not in t1_map:
        vr.errors.append("Table 1 has no mappable 'Report Name' column (required as identity/join key).")
        return vr

    # Table 2 is optional but, if present, must have a report name.
    if t2 is None or len(t2) == 0:
        vr.warnings.append("Table 2 (execution) missing/empty — all reports flagged 'not in execution table'.")
    elif "t2_report_name" not in t2_map:
        vr.warnings.append("Table 2 present but has no mappable 'Custom Report' column — execution join skipped.")

    # Scoring-field coverage (warn on missing signals).
    for fkey in schema.SCORING_FIELDS:
        if fkey not in t1_map:
            vr.warnings.append(f"Scoring field '{fkey}' not found in Table 1 — that signal scores as absent.")

    vr.field_coverage = coverage(t1.rename(columns={v: k for k, v in t1_map.items()}))
    return vr
