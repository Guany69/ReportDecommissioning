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
from .hard_rules import HARD_RULE_RECOMMENDATION
from .report_fields import (calculate_field_containment,
                            calculate_field_similarity)
from .soft_scoring import Reason

_META_FIELDS = ("description", "category", "report_tag", "data_source",
                "report_prompts", "areas_used", "landing_page", "owner")

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


def _metadata_completeness(m: dict) -> int:
    return sum(1 for k in _META_FIELDS
               if m.get(k) is not None and not pd.isna(m.get(k)) and str(m.get(k)).strip())


def _active_owner(m: dict) -> bool:
    """Owner present and not flagged inactive."""
    owner = m.get("owner")
    if owner is None or pd.isna(owner) or not str(owner).strip():
        return False
    return "Inactive Owner" not in (m.get("ownership_flags") or [])


def _shared_yes(m: dict) -> bool:
    v = m.get("shared")
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return False
    return bool(v) if not isinstance(v, str) else is_true(v)


def _has(m: dict, key: str) -> bool:
    v = m.get(key)
    return v is not None and not pd.isna(v) and bool(str(v).strip())


def choose_suggested_keeper(group_members: list[dict]) -> tuple[int, str]:
    """Return (keeper_uid, consolidation_reason).

    Priority (highest wins): field coverage, lowest risk, most recent run,
    highest run count, active owner, metadata completeness, Shared=Yes, has
    areas-where-used, has landing page; ties broken by lower report_uid.
    """
    def key(m):
        run = _ts(m.get("last_run_date"))
        run_rank = run.value if run is not None else -1
        risk = _num(m.get("total_risk_score"))
        risk = risk if risk is not None else 1e9   # missing risk = worst
        return (
            _effective_field_count(m),   # 0 when field extraction unavailable (neutral)
            -risk,
            run_rank,
            _num(m.get("times_run")) or -1,
            int(_active_owner(m)),
            _metadata_completeness(m),
            int(_shared_yes(m)),
            int(_has(m, "areas_used")),
            int(_has(m, "landing_page")),
            -m["report_uid"],
        )

    keeper = max(group_members, key=key)
    reason = (f"Suggested keeper because it has the most report fields "
              f"({_effective_field_count(keeper)}) and the lowest risk score "
              f"({int(_num(keeper.get('total_risk_score')) or 0)}) in its group.")
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

    `group` is the DupGroup the report belongs to (or None).
    """
    if r.get("is_hard_rule"):
        return HARD_RULE_RECOMMENDATION

    band = r.get("recommendation") or r.get("band") or "Keep"

    if group is None:
        # Report is not in a confirmed duplicate group but may still have been
        # flagged as metadata-similar with missing field evidence.
        if r.get("duplicate_classification") == "Metadata Similar - Fields Unavailable":
            return "Needs Manual Review - Field Comparison Missing"
        return band

    if r["report_uid"] == group.keeper_uid:
        return "Keep"

    classification = r.get("duplicate_classification", "")
    if classification in _STRONG:
        return "Delete Duplicate After Migration"
    if classification in _CONSOLIDATE:
        return "Consolidate After Review"
    if classification == "Metadata Similar - Fields Unavailable":
        return "Needs Manual Review - Field Comparison Missing"
    return "Needs Manual Review"
