"""Report Fields parsing + set comparison for Stage-2 duplicate analysis.

A report's "Report Fields" column lists the columns inside it, e.g.
    "Employee ID, Employee Name, Manager, Location, Hire Date"
We parse it into a normalized set, then compare two reports with Jaccard
similarity and containment. This is only ever invoked for pairs that already
passed the Stage-1 metadata gate (see dedup.should_compare_report_fields).
"""
from __future__ import annotations

import re

from .clean import text

# Separators a source might use between field names.
_SPLIT = re.compile(r"[,;|\n\r]+")


def normalize_report_field(name: str, aliases: dict[str, str] | None = None) -> str:
    """Lowercase, trim, underscores->spaces, collapse spaces, strip punctuation,
    then apply the alias map (e.g. 'Worker ID' -> 'employee id')."""
    s = text(name).lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if aliases and s in aliases:
        s = aliases[s]
    return s


def parse_report_fields(report_fields_text, aliases: dict[str, str] | None = None) -> set[str]:
    """Split on comma/semicolon/pipe/newline into a set of normalized fields."""
    raw = text(report_fields_text)
    if not raw:
        return set()
    out = set()
    for part in _SPLIT.split(raw):
        norm = normalize_report_field(part, aliases)
        if norm:
            out.add(norm)
    return out


def calculate_field_similarity(fields_a: set[str], fields_b: set[str]) -> float | None:
    """Jaccard similarity (percent) = shared / total_unique. None if no data."""
    if not fields_a or not fields_b:
        return None
    shared = len(fields_a & fields_b)
    union = len(fields_a | fields_b)
    return round(100.0 * shared / union, 1) if union else None


def calculate_field_containment(fields_a: set[str], fields_b: set[str]) -> float | None:
    """Containment (percent) = shared / smaller-report field count. None if no data."""
    if not fields_a or not fields_b:
        return None
    shared = len(fields_a & fields_b)
    smaller = min(len(fields_a), len(fields_b))
    return round(100.0 * shared / smaller, 1) if smaller else None
