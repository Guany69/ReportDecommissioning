"""Validation: warn where possible, abort only when the engine cannot proceed."""
from __future__ import annotations

from dataclasses import dataclass, field

from collections import Counter

import pandas as pd

from . import schema
from .clean import normalize_report_name


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_coverage: dict[str, float] = field(default_factory=dict)
    duplicate_report_names: list[str] = field(default_factory=list)

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

    # Normalized Custom Report names must be unique — duplicates would let two
    # unrelated rows collide on the exact join key. Surface clearly; do not merge.
    name_col = t1_map["report_name"]
    keys = [normalize_report_name(v) for v in t1[name_col].tolist()]
    dup_counts = {k: c for k, c in Counter(k for k in keys if k).items() if c > 1}
    if dup_counts:
        sample = ", ".join(sorted(dup_counts)[:5])
        more = "" if len(dup_counts) <= 5 else f" (+{len(dup_counts) - 5} more)"
        vr.warnings.append(
            f"Table 1 has {len(dup_counts)} duplicate normalized 'Custom Report' "
            f"name(s) — execution data may be attributed to multiple rows: {sample}{more}."
        )
        vr.duplicate_report_names = sorted(dup_counts)

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
