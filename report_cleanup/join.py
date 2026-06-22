"""Join Comprehensive (Table 1) metadata to the Runs (Table 2) execution rollup.

Matching is an EXACT normalized-report-name lookup (clean.normalize_report_name):
case-insensitive, whitespace-trimmed, not fuzzy. The two exports are guaranteed
to share report names, and Comprehensive names are validated unique upstream
(validate.py), so each report maps to at most one rollup entry.
"""
from __future__ import annotations

import pandas as pd

from .clean import normalize_report_name


def _max_ts(*values):
    """NaT-safe max of timestamp-ish values; returns pd.NaT when all are blank."""
    cand = []
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            continue
        cand.append(pd.Timestamp(v))
    return max(cand) if cand else pd.NaT


def join_reports(t1: pd.DataFrame, rollup: dict, name_noise=None) -> tuple[list[dict], dict]:
    """Return (records, diagnostics). Each record is the metadata row plus the
    Runs execution rollup fields. ``name_noise`` is accepted for call-site
    compatibility but unused (the key is exact)."""
    diag = {"matched": 0, "unmatched": 0}

    records: list[dict] = []
    for i, row in t1.reset_index(drop=True).iterrows():
        rec = {k: row[k] for k in t1.columns}
        rec["report_uid"] = int(i)

        nk = normalize_report_name(rec.get("report_name"))
        match = rollup.get(nk)

        if match:
            diag["matched"] += 1
            rec["runs_exec_count"] = match["runs_exec_count"]
            rec["latest_runs_start"] = match["latest_runs_start"]
            rec["latest_runs_last_run"] = match["latest_runs_last_run"]
            rec["latest_runs_execution"] = match["latest_runs_execution"]
            rec["distinct_requesters"] = match["distinct_requesters"]
            rec["exec_modes"] = match["exec_modes"]
            rec["runs_start_timestamps"] = match["start_timestamps"]
            rec["invalid_run_timestamps"] = match.get("unparsed_start_count", 0) > 0
            rec["t2_owner"] = match.get("t2_owner", "")
            rec["appears_in_exec"] = match["runs_exec_count"] > 0
        else:
            diag["unmatched"] += 1
            rec["runs_exec_count"] = 0
            rec["latest_runs_start"] = pd.NaT
            rec["latest_runs_last_run"] = pd.NaT
            rec["latest_runs_execution"] = pd.NaT
            rec["distinct_requesters"] = 0
            rec["exec_modes"] = ""
            rec["runs_start_timestamps"] = []
            rec["invalid_run_timestamps"] = False
            rec["t2_owner"] = ""
            rec["appears_in_exec"] = False

        # Comprehensive long-term execution count is kept separate from the
        # windowed Runs count — never compared against it.
        rec["comprehensive_exec_count"] = rec.get("times_run")

        # Legacy field names retained for DB/Excel column compatibility.
        rec["exec_count"] = rec["runs_exec_count"]
        rec["last_exec_date"] = rec["latest_runs_execution"]

        records.append(rec)

    # Legacy diagnostic keys (composite/name_only/ambiguous) preserved so the DB
    # schema and existing displays keep working; the join is now name-exact, so
    # every match is reported under "composite" and the fuzzy buckets are zero.
    diag["composite"] = diag["matched"]
    diag["name_only"] = 0
    diag["ambiguous"] = 0

    return records, diag


def derive_effective_last_run(records: list[dict]) -> list[dict]:
    """effective_last_run_date = max(Comprehensive Last Run Date, latest Runs execution).

    Also stamps comprehensive_last_run_date and a runs_newer flag so reviewers can
    see when the six-month Runs history superseded a stale Comprehensive date.
    """
    for r in records:
        comp = r.get("last_run_date")
        runs_exec = r.get("latest_runs_execution", pd.NaT)
        eff = _max_ts(comp, runs_exec)

        r["comprehensive_last_run_date"] = comp if (comp is not None and not pd.isna(comp)) else pd.NaT
        r["effective_last_run_date"] = eff

        comp_blank = comp is None or pd.isna(comp)
        runs_blank = runs_exec is None or pd.isna(runs_exec)
        r["runs_newer_than_comprehensive"] = bool(
            not runs_blank and (comp_blank or pd.Timestamp(runs_exec) > pd.Timestamp(comp))
        )

        # Legacy alias used by the DB layer.
        r["eff_last_run"] = eff
    return records
