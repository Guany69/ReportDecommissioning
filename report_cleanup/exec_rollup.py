"""Collapse the Runs export (many execution rows per report) to one rollup per report.

Reports are keyed by the EXACT normalized report name (clean.normalize_report_name)
because Comprehensive.Custom Report and Runs.Report Name are guaranteed to carry
identical names. We deliberately do NOT key on type/owner: a single report's runs
must all aggregate together even when the Runs owner/type differs from Comprehensive
(owner drift is itself surfaced as a flag elsewhere).

The Runs export covers roughly six months, so its row count is a windowed metric
and must never be compared against the Comprehensive long-term execution count.
"""
from __future__ import annotations

import pandas as pd

from .clean import normalize_report_name, text


def _valid_ts(v):
    """Return a Timestamp if v is a real date, else None."""
    if v is None or pd.isna(v):
        return None
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts


def build_exec_rollup(t2: pd.DataFrame, name_noise=None) -> dict:
    """Return {name_key: rollup} aggregating every Runs row for that report name.

    ``name_noise`` is accepted for call-site compatibility but intentionally
    unused — the join key is exact, not de-noised.

    Each rollup entry contains:
      runs_exec_count        total Runs rows (windowed ~6 months)
      latest_runs_start      max valid Start Date and Time
      latest_runs_last_run   max valid Runs Last Run Date
      latest_runs_execution  max(latest_runs_start, latest_runs_last_run)
      distinct_requesters    count of distinct non-blank requesters
      exec_modes             comma-joined sorted execution modes
      start_timestamps       sorted list of valid Start Date timestamps (recurrence)
      t2_owner               an observed Runs owner (for owner-mismatch flagging)
    """
    if t2 is None or len(t2) == 0:
        return {}

    has = lambda c: c in t2.columns
    agg: dict[str, dict] = {}

    for _, row in t2.iterrows():
        name = row.get("t2_report_name") if has("t2_report_name") else None
        nk = normalize_report_name(name)
        if not nk:
            continue

        owner = row.get("t2_report_owner") if has("t2_report_owner") else None
        start = _valid_ts(row.get("t2_start_date")) if has("t2_start_date") else None
        last_run = _valid_ts(row.get("t2_last_run_date")) if has("t2_last_run_date") else None
        # Requester breadth uses the human name; fall back to the employee id.
        requester = None
        if has("t2_requested_by"):
            requester = text(row.get("t2_requested_by")) or None
        if requester is None and has("t2_requested_id"):
            requester = text(row.get("t2_requested_id")) or None
        mode = row.get("t2_exec_mode") if has("t2_exec_mode") else None

        a = agg.setdefault(nk, {
            "runs_exec_count": 0,
            "start_dates": [],
            "last_run_dates": [],
            "requesters": set(),
            "modes": set(),
            "owner": text(owner),
            "unparsed_start_count": 0,
        })
        a["runs_exec_count"] += 1
        if start is not None:
            a["start_dates"].append(start)
        elif has("t2_start_date"):
            # A Runs row with no usable Start timestamp (blank or unparseable):
            # excluded from recurrence, but the row still counts as a run.
            a["unparsed_start_count"] += 1
        if last_run is not None:
            a["last_run_dates"].append(last_run)
        if requester:
            a["requesters"].add(requester)
        if mode is not None and not pd.isna(mode):
            m = text(mode)
            if m:
                a["modes"].add(m)
        if not a["owner"] and text(owner):
            a["owner"] = text(owner)

    rollup: dict[str, dict] = {}
    for nk, a in agg.items():
        latest_start = max(a["start_dates"]) if a["start_dates"] else pd.NaT
        latest_last_run = max(a["last_run_dates"]) if a["last_run_dates"] else pd.NaT
        candidates = [d for d in (latest_start, latest_last_run) if not pd.isna(d)]
        latest_exec = max(candidates) if candidates else pd.NaT
        rollup[nk] = {
            "runs_exec_count": a["runs_exec_count"],
            "latest_runs_start": latest_start,
            "latest_runs_last_run": latest_last_run,
            "latest_runs_execution": latest_exec,
            "distinct_requesters": len(a["requesters"]),
            "exec_modes": ", ".join(sorted(a["modes"])),
            "start_timestamps": sorted(a["start_dates"]),
            "unparsed_start_count": a["unparsed_start_count"],
            "t2_owner": a["owner"],
        }
    return rollup
