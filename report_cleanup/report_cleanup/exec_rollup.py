"""Collapse Table 2 (many usage rows per report) to one row per report key.

Produces both a composite-key rollup (name|type|owner) and a name-only rollup
so join.py can fall back when Type/Owner are blank or differ across tables.
"""
from __future__ import annotations

import pandas as pd

from .clean import normalize_name, text


def _key(name, rtype, owner, name_noise) -> str:
    return "|".join([
        normalize_name(name, name_noise),
        text(rtype).lower(),
        text(owner).lower(),
    ])


def _name_key(name, name_noise) -> str:
    return normalize_name(name, name_noise)


def build_exec_rollup(t2: pd.DataFrame, name_noise) -> dict:
    """Return {'composite': {key: rollup}, 'name': {name_key: rollup_or_AMBIGUOUS}}.

    A name-only key that maps to more than one distinct composite group is marked
    ambiguous (None) so the join never guesses which report the usage belongs to.
    """
    composite: dict[str, dict] = {}
    name_groups: dict[str, set] = {}

    if t2 is None or len(t2) == 0:
        return {"composite": {}, "name": {}}

    has = lambda c: c in t2.columns
    for _, row in t2.iterrows():
        name = row.get("t2_report_name") if has("t2_report_name") else None
        rtype = row.get("t2_report_type") if has("t2_report_type") else None
        owner = row.get("t2_report_owner") if has("t2_report_owner") else None
        start = row.get("t2_start_date") if has("t2_start_date") else pd.NaT
        req_id = row.get("t2_requested_id") if has("t2_requested_id") else None
        mode = row.get("t2_exec_mode") if has("t2_exec_type") else None

        ck = _key(name, rtype, owner, name_noise)
        nk = _name_key(name, name_noise)

        agg = composite.setdefault(ck, {
            "exec_count": 0, "dates": [], "requesters": set(), "modes": set(),
            "owner": text(owner),
        })
        agg["exec_count"] += 1
        if not pd.isna(start):
            agg["dates"].append(start)
        if not (req_id is None or pd.isna(req_id)):
            agg["requesters"].add(str(req_id))
        if not (mode is None or pd.isna(mode)):
            agg["modes"].add(str(mode))

        name_groups.setdefault(nk, set()).add(ck)

    # Finalize composite rollups.
    comp_final = {}
    for ck, agg in composite.items():
        dates = agg["dates"]
        comp_final[ck] = {
            "exec_count": agg["exec_count"],
            "last_exec_date": max(dates) if dates else pd.NaT,
            "first_exec_date": min(dates) if dates else pd.NaT,
            "distinct_requesters": len(agg["requesters"]),
            "exec_modes": ", ".join(sorted(agg["modes"])),
            "t2_owner": agg["owner"],
        }

    # name-only rollup: only usable when it maps to exactly one composite group.
    name_final = {}
    for nk, cks in name_groups.items():
        if len(cks) == 1:
            name_final[nk] = comp_final[next(iter(cks))]
        else:
            name_final[nk] = None  # ambiguous

    return {"composite": comp_final, "name": name_final}
