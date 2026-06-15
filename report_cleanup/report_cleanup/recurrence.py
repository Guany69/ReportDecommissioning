"""Recurring-execution detection from execution TIMESTAMPS ONLY.

Execution Mode / Execution Type / Requested By are deliberately ignored — a
report is "recurring" only if its execution timestamps show a regular cadence.
The Runs export spans ~6 months, so absence of an annual/quarterly cadence is
never treated as evidence that a report is non-recurring.

Pipeline supplies each report's valid Runs Start Date timestamps (primary). The
detector sorts them, collapses bursts (retries / parameter changes) within a
configurable window into single occurrences, measures inter-occurrence intervals,
and matches them against configurable cadence ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CLASS_STRONG = "Strong"
CLASS_MODERATE = "Moderate"
CLASS_INSUFFICIENT = "Insufficient History"
CLASS_NONE = "No Recurring Pattern"


@dataclass
class RecurrenceResult:
    classification: str
    cadence: str | None
    match_percentage: float          # fraction of intervals fitting `cadence` (0..1)
    occurrences: int                 # distinct occurrences after burst collapsing
    intervals: int
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None

    def as_dict(self) -> dict:
        return {
            "classification": self.classification,
            "cadence": self.cadence,
            "match_percentage": self.match_percentage,
            "occurrences": self.occurrences,
            "intervals": self.intervals,
            "earliest": None if self.earliest is None else pd.Timestamp(self.earliest),
            "latest": None if self.latest is None else pd.Timestamp(self.latest),
        }


def _valid_sorted(timestamps) -> list[pd.Timestamp]:
    out = []
    for t in timestamps or []:
        if t is None:
            continue
        try:
            if pd.isna(t):
                continue
            ts = pd.Timestamp(t)
        except (TypeError, ValueError):
            continue
        if not pd.isna(ts):
            out.append(ts)
    return sorted(out)


def _collapse_bursts(sorted_ts: list[pd.Timestamp], window_minutes: float) -> list[pd.Timestamp]:
    """Collapse runs within `window_minutes` of the current occurrence into one.

    The first timestamp of each burst represents the occurrence.
    """
    if not sorted_ts:
        return []
    window = pd.Timedelta(minutes=window_minutes)
    occurrences = [sorted_ts[0]]
    for ts in sorted_ts[1:]:
        if ts - occurrences[-1] > window:
            occurrences.append(ts)
    return occurrences


def _best_cadence(interval_days: list[float], cadences: dict) -> tuple[str | None, float]:
    """Return (cadence_name, fraction_of_intervals_in_range) for the best cadence."""
    if not interval_days:
        return None, 0.0
    best_name, best_frac = None, 0.0
    for name, rng in cadences.items():
        lo, hi = rng["min_days"], rng["max_days"]
        hits = sum(1 for d in interval_days if lo <= d <= hi)
        frac = hits / len(interval_days)
        if frac > best_frac:
            best_name, best_frac = name, frac
    return best_name, best_frac


def detect_recurrence(timestamps, cfg) -> RecurrenceResult:
    rc = cfg.recurrence
    window = rc.get("burst_window_minutes", 30)
    cadences = rc.get("cadences", {})

    sorted_ts = _valid_sorted(timestamps)
    occurrences = _collapse_bursts(sorted_ts, window)
    n = len(occurrences)
    earliest = occurrences[0] if occurrences else None
    latest = occurrences[-1] if occurrences else None

    moderate_min = rc.get("moderate_min_occurrences", 3)
    if n < moderate_min:
        return RecurrenceResult(CLASS_INSUFFICIENT, None, 0.0, n, max(0, n - 1), earliest, latest)

    interval_days = [
        (occurrences[i + 1] - occurrences[i]).total_seconds() / 86400.0
        for i in range(len(occurrences) - 1)
    ]
    cadence, frac = _best_cadence(interval_days, cadences)

    strong_min_occ = rc.get("strong_min_occurrences", 4)
    strong_min_int = rc.get("strong_min_intervals", 3)
    strong_thresh = rc.get("strong_match_threshold", 0.75)

    if (n >= strong_min_occ and len(interval_days) >= strong_min_int and frac >= strong_thresh):
        return RecurrenceResult(CLASS_STRONG, cadence, frac, n, len(interval_days), earliest, latest)

    # Moderate: just enough history and every interval agrees on one cadence.
    if frac >= 1.0 and len(interval_days) >= 2:
        return RecurrenceResult(CLASS_MODERATE, cadence, frac, n, len(interval_days), earliest, latest)

    return RecurrenceResult(CLASS_NONE, None, frac, n, len(interval_days), earliest, latest)
