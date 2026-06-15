"""Join Table 1 metadata to the Table 2 execution rollup.

Composite key (name|type|owner) first; name-only fallback only when that name
maps to exactly one execution group. Builds the list of report records the rest
of the pipeline operates on.
"""
from __future__ import annotations

import pandas as pd

from .clean import normalize_name, text


def _composite_key(name, rtype, owner, name_noise) -> str:
    return "|".join([
        normalize_name(name, name_noise),
        text(rtype).lower(),
        text(owner).lower(),
    ])


def join_reports(t1: pd.DataFrame, rollup: dict, name_noise) -> tuple[list[dict], dict]:
    """Return (records, diagnostics). Each record is a dict with metadata +
    execution rollup fields + an effective last-run date."""
    comp = rollup["composite"]
    name_idx = rollup["name"]
    diag = {"composite": 0, "name_only": 0, "unmatched": 0, "ambiguous": 0}

    records: list[dict] = []
    for i, row in t1.reset_index(drop=True).iterrows():
        rec = {k: row[k] for k in t1.columns}
        rec["report_uid"] = int(i)

        name = rec.get("report_name")
        rtype = rec.get("report_type")
        owner = rec.get("owner")
        ck = _composite_key(name, rtype, owner, name_noise)
        nk = normalize_name(name, name_noise)

        match = None
        if ck in comp:
            match = comp[ck]
            diag["composite"] += 1
        elif nk in name_idx:
            if name_idx[nk] is None:
                diag["ambiguous"] += 1
            else:
                match = name_idx[nk]
                diag["name_only"] += 1

        if match:
            rec["exec_count"] = match["exec_count"]
            rec["last_exec_date"] = match["last_exec_date"]
            rec["distinct_requesters"] = match["distinct_requesters"]
            rec["exec_modes"] = match["exec_modes"]
            rec["t2_owner"] = match.get("t2_owner", "")
            rec["appears_in_exec"] = match["exec_count"] > 0
        else:
            rec["exec_count"] = 0
            rec["last_exec_date"] = pd.NaT
            rec["distinct_requesters"] = 0
            rec["exec_modes"] = ""
            rec["t2_owner"] = ""
            rec["appears_in_exec"] = False
            diag["unmatched"] += 1

        records.append(rec)

    return records, diag


def derive_effective_last_run(records: list[dict]) -> list[dict]:
    """eff_last_run = COALESCE(metadata.last_run_date, last_exec_date)."""
    for r in records:
        meta = r.get("last_run_date")
        if meta is not None and not pd.isna(meta):
            r["eff_last_run"] = meta
        else:
            r["eff_last_run"] = r.get("last_exec_date", pd.NaT)
    return records
