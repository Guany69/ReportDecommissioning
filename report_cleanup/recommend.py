"""Keeper selection, per-member duplicate analysis, and suggested actions.

Risk scoring already decided "how cleanup-worthy is this report?" (the band /
recommendation). This module answers the *duplicate* question: within a group of
similar reports, which one should we keep, and what should happen to the rest?

Nothing here deletes anything — every output is a recommendation.
"""
from __future__ import annotations

import pandas as pd

from .clean import is_true
from .dedup import (calculate_name_similarity, candidate_similarity_score,
                    classify_field_match)
from .protection import _distinct_area_count
from .report_fields import (calculate_field_containment,
                            calculate_field_similarity)
from .soft_scoring import Reason

def recommend_label(overall_score, cfg) -> str:
    """Map an Overall Decommissioning Score (0..100) to its recommendation label
    using the configurable, high-to-low thresholds in cfg.recommendation."""
    if overall_score is None:
        return "Keep"
    thresholds = sorted(cfg.recommendation["thresholds"], key=lambda t: t["min"], reverse=True)
    for t in thresholds:
        if overall_score >= t["min"]:
            return t["label"]
    return thresholds[-1]["label"]


_STRONG = {"Nearly Identical Duplicate", "One Report Contained In Another"}
_CONSOLIDATE = {"Strong Duplicate Candidate", "Strong Consolidation Candidate"}


def _num(v):
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts(v):
    return None if (v is None or pd.isna(v)) else pd.Timestamp(v)


def _field_count(m: dict) -> int:
    return len(m.get("report_fields_set") or set())


def _effective_field_count(m: dict) -> int:
    """Field count used for keeper ranking.

    Only counts fields when extraction was confirmed — prevents reports
    with unavailable field data from being penalised (they score neutral 0)
    rather than being mistakenly ranked below reports with confirmed-zero fields.
    """
    if m.get("field_extraction_status") == "Matched":
        return _field_count(m)
    return 0


def _shared_yes(m: dict) -> bool:
    v = m.get("shared")
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return False
    return bool(v) if not isinstance(v, str) else is_true(v)


_RECURRENCE_RANK = {"Strong": 3, "Moderate": 2, "No Recurring Pattern": 1, "Insufficient History": 0}


def _overall(m: dict) -> float:
    """Overall Decommissioning Score, falling back to cleanup risk for callers that
    only set the legacy field."""
    v = _num(m.get("overall_score"))
    if v is None:
        v = _num(m.get("total_risk_score"))
    return v if v is not None else 1e9


def _recurrence_rank(m: dict) -> int:
    return _RECURRENCE_RANK.get(str((m.get("recurrence") or {}).get("classification", "")), 0)


def choose_suggested_keeper(group_members: list[dict]) -> tuple[int, str]:
    """Return (keeper_uid, keeper_reason).

    Priority (highest wins), per requirement: lower Overall Score, more recent
    effective last run, stronger recurrence, more distinct requesters, greater
    field coverage, more Areas Where Used, Shared=Yes, more recent Last Updated,
    then a stable lower-report_uid tiebreaker. NOT chosen on field coverage alone.
    """
    def key(m):
        eff = _ts(m.get("effective_last_run_date")) or _ts(m.get("last_run_date"))
        eff_rank = eff.value if eff is not None else -1
        lu = _ts(m.get("last_updated"))
        lu_rank = lu.value if lu is not None else -1
        return (
            -_overall(m),                          # lower Overall Score wins
            eff_rank,                              # more recent effective last run
            _recurrence_rank(m),                   # stronger recurrence
            _num(m.get("distinct_requesters")) or 0,
            _effective_field_count(m),             # field coverage (not the primary signal)
            _distinct_area_count(m),               # more Areas Where Used
            int(_shared_yes(m)),
            lu_rank,                               # more recent Last Updated
            -m["report_uid"],                      # stable tiebreak
        )

    keeper = max(group_members, key=key)
    reason = (f"Recommended keeper: lowest Overall Decommissioning Score "
              f"({int(_overall(keeper)) if _overall(keeper) < 1e9 else 'n/a'}) with the most "
              f"recent effective last run in its group.")
    return keeper["report_uid"], reason


def apply_duplicate_analysis(records_by_uid: dict[int, dict], groups, cfg) -> None:
    """Pick a keeper per group and attach every member's vs-keeper duplicate stats."""
    name_noise = cfg.clean["name_noise"]
    for g in groups:
        members = [records_by_uid[u] for u in g.members]
        g.keeper_uid, keeper_reason = choose_suggested_keeper(members)
        keeper = records_by_uid[g.keeper_uid]
        keeper_fields = keeper.get("report_fields_set") or set()
        keeper_name = str(keeper.get("report_name") or g.keeper_uid)

        for m in members:
            m["dup_group_id"] = g.group_id
            m["suggested_keeper_report_name"] = keeper_name
            m["field_count"] = _field_count(m)
            is_keeper = (m["report_uid"] == g.keeper_uid)
            m["is_suggested_keeper"] = is_keeper

            mf = m.get("report_fields_set") or set()
            if is_keeper:
                m["shared_field_count"] = len(mf)
                m["field_similarity_percent"] = 100.0 if mf else None
                m["field_containment_percent"] = 100.0 if mf else None
                m["missing_fields_compared_to_keeper"] = []
                m["extra_fields_in_keeper"] = []
                m["report_name_similarity_percent"] = 100.0
                m["candidate_similarity_score"] = 100.0
                m["duplicate_classification"] = "Suggested Keeper"
                m["consolidation_reason"] = keeper_reason
                m.setdefault("dup_reasons", []).append(Reason(
                    "duplicate",
                    f"Suggested keeper of {g.group_id} ({len(members)} similar reports). {keeper_reason}",
                    None))
                continue

            sim = calculate_field_similarity(mf, keeper_fields)
            cont = calculate_field_containment(mf, keeper_fields)
            shared = sorted(mf & keeper_fields)
            only_here = sorted(mf - keeper_fields)
            only_keeper = sorted(keeper_fields - mf)
            m["shared_field_count"] = len(shared)
            m["field_similarity_percent"] = sim
            m["field_containment_percent"] = cont
            m["missing_fields_compared_to_keeper"] = only_here
            m["extra_fields_in_keeper"] = only_keeper
            m["report_name_similarity_percent"] = calculate_name_similarity(
                m.get("report_name"), keeper_name, name_noise)
            m["candidate_similarity_score"] = candidate_similarity_score(m, keeper, cfg)
            classification = classify_field_match(sim, cont, cfg)
            m["duplicate_classification"] = classification
            reason = _member_reason(classification, keeper_name)
            m["consolidation_reason"] = reason

            # Reason trail: spell out the "copy of X" verdict and the field overlap.
            dup = m.setdefault("dup_reasons", [])
            match_bits = []
            if sim is not None:
                match_bits.append(f"{sim:.0f}% field match")
            if cont is not None:
                match_bits.append(f"{cont:.0f}% contained in keeper")
            match_txt = f" ({', '.join(match_bits)})" if match_bits else ""
            dup.append(Reason("duplicate",
                              f'Possible copy of "{keeper_name}" — {classification}{match_txt}', None))
            if shared:
                dup.append(Reason("fields",
                                  f"Matching fields ({len(shared)}): {', '.join(shared)}", None))
            if only_here:
                dup.append(Reason("fields", f"Only in this report: {', '.join(only_here)}", None))
            if only_keeper:
                dup.append(Reason("fields", f"Only in keeper: {', '.join(only_keeper)}", None))


def _member_reason(classification: str, keeper_name: str) -> str:
    if classification == "One Report Contained In Another":
        return "Suggested duplicate removal because this report is fully contained in the keeper."
    if classification in _STRONG:
        return f'Suggested duplicate removal because its fields closely match "{keeper_name}".'
    if classification in _CONSOLIDATE:
        return f'Suggested consolidation into "{keeper_name}" — field sets largely overlap.'
    return "Needs review because fields are similar but neither report fully contains the other."


def resolve_recommendation(r: dict, group, records_by_uid: dict[int, dict], cfg) -> str:
    """Return the suggested_action for one report.

    Built from the Overall-Score recommendation label, with duplicate flags
    adjusting the WORDING only — never the numeric score (`group` is the DupGroup
    the report belongs to, or None).
    """
    base = r.get("recommendation") or r.get("band") or "Keep"

    # Is this report duplicate-flagged? Either by the weighted model or by being a
    # non-keeper member of a confirmed field-duplicate group.
    in_group_non_keeper = group is not None and r["report_uid"] != group.keeper_uid
    is_dup = bool(r.get("potential_duplicate")) or in_group_non_keeper

    if r.get("is_hard_rule"):
        return f"{base}; duplicate keeper identified" if is_dup else base

    if not is_dup:
        # Surface a metadata-similar-but-unconfirmed report for manual review.
        if r.get("duplicate_classification") == "Metadata Similar - Fields Unavailable":
            return "Needs Manual Review - Field Comparison Missing"
        return base

    overall = r.get("overall_score")
    if overall is not None and overall >= 60:
        return "Consolidation Review"
    if overall is not None and overall < 40:
        return "Duplicate Consolidation Review; active report may be the keeper"
    return f"{base}; possible duplicate"
