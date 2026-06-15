"""Keeper selection (new priority) and duplicate-aware recommendation wording."""
import pandas as pd

from report_cleanup.dedup import DupGroup
from report_cleanup.recommend import (apply_duplicate_analysis,
                                       choose_suggested_keeper,
                                       resolve_recommendation)

TODAY = pd.Timestamp("2026-06-05")


def rec(uid, fields="", overall=10, recurrence="Insufficient History",
        last_run_days=10, **kw):
    s = {f.strip() for f in fields.split(",") if f.strip()}
    r = dict(report_uid=uid, report_name=f"R{uid}", report_fields_set=s,
             field_extraction_status="Matched" if s else "No fields found",
             overall_score=overall, total_risk_score=overall,
             effective_last_run_date=TODAY - pd.Timedelta(days=last_run_days),
             last_run_date=TODAY - pd.Timedelta(days=last_run_days),
             last_updated=TODAY, recurrence={"classification": recurrence},
             distinct_requesters=0, times_run=10, owner="o", ownership_flags=[],
             shared="Yes", areas_used="a", landing_page="l", description="d",
             category="c", report_tag="t", data_source="ds", report_prompts="p",
             is_hard_rule=False, recommendation="Keep")
    r.update(kw)
    return r


# ---- Keeper selection ------------------------------------------------------
def test_keeper_prefers_lower_overall_score_not_field_coverage(cfg):
    rich = rec(0, "a,b,c,d,e", overall=40)   # most fields but higher score
    lean = rec(1, "a,b,c", overall=5)        # fewer fields, lower score -> keeper
    keeper_uid, reason = choose_suggested_keeper([rich, lean])
    assert keeper_uid == 1
    assert "lowest Overall Decommissioning Score" in reason


def test_keeper_breaks_overall_tie_on_recency(cfg):
    a = rec(0, "a,b,c", overall=10, last_run_days=100)
    b = rec(1, "a,b,c", overall=10, last_run_days=5)   # more recent -> keeper
    keeper_uid, _ = choose_suggested_keeper([a, b])
    assert keeper_uid == 1


def test_keeper_breaks_recency_tie_on_recurrence(cfg):
    a = rec(0, "a,b,c", overall=10, last_run_days=5, recurrence="Insufficient History")
    b = rec(1, "a,b,c", overall=10, last_run_days=5, recurrence="Strong")  # stronger -> keeper
    keeper_uid, _ = choose_suggested_keeper([a, b])
    assert keeper_uid == 1


# ---- Per-member duplicate columns ------------------------------------------
def test_contained_member_columns_and_keeper(cfg):
    keeper = rec(0, "a,b,c,d,e", overall=10)
    contained = rec(1, "a,b,c", overall=40)          # fully inside keeper, higher score
    by_uid = {0: keeper, 1: contained}
    g = DupGroup("DUP-0001", [0, 1])
    apply_duplicate_analysis(by_uid, [g], cfg)
    assert g.keeper_uid == 0
    assert keeper["is_suggested_keeper"] is True
    assert contained["field_containment_percent"] == 100.0
    assert contained["duplicate_classification"] == "One Report Contained In Another"
    assert set(contained["extra_fields_in_keeper"]) == {"d", "e"}
    assert contained["consolidation_reason"]


# ---- resolve_recommendation: duplicate flags change WORDING only -----------
def test_hard_rule_with_duplicate_notes_keeper(cfg):
    r = rec(0, is_hard_rule=True, recommendation="Automatic Decommissioning Recommendation",
            potential_duplicate=True)
    assert resolve_recommendation(r, None, {0: r}, cfg) == \
        "Automatic Decommissioning Recommendation; duplicate keeper identified"


def test_hard_rule_without_duplicate_is_plain(cfg):
    r = rec(0, is_hard_rule=True, recommendation="Automatic Decommissioning Recommendation")
    assert resolve_recommendation(r, None, {0: r}, cfg) == "Automatic Decommissioning Recommendation"


def test_high_score_duplicate_is_consolidation_review(cfg):
    r = rec(0, overall=72, potential_duplicate=True, recommendation="Decommissioning Review")
    assert resolve_recommendation(r, None, {0: r}, cfg) == "Consolidation Review"


def test_low_score_duplicate_flags_active_keeper(cfg):
    r = rec(0, overall=25, potential_duplicate=True, recommendation="Keep")
    assert resolve_recommendation(r, None, {0: r}, cfg) == \
        "Duplicate Consolidation Review; active report may be the keeper"


def test_non_duplicate_returns_base_label(cfg):
    r = rec(0, overall=20, recommendation="Keep")
    assert resolve_recommendation(r, None, {0: r}, cfg) == "Keep"


def test_duplicate_flag_does_not_change_overall_score(cfg):
    r = rec(0, overall=50, potential_duplicate=True, recommendation="Owner Review / Monitor")
    before = r["overall_score"]
    resolve_recommendation(r, None, {0: r}, cfg)
    assert r["overall_score"] == before  # recommendation wording never mutates the score
