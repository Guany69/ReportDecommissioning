"""Keeper selection, duplicate classification, and suggested actions."""
import pandas as pd

from report_cleanup.dedup import DupGroup
from report_cleanup.recommend import (apply_duplicate_analysis,
                                       choose_suggested_keeper,
                                       resolve_recommendation)

TODAY = pd.Timestamp("2026-06-05")


def rec(uid, fields="", risk=10, **kw):
    s = {f.strip() for f in fields.split(",") if f.strip()}
    # field_extraction_status="Matched" so _effective_field_count uses the real field set.
    r = dict(report_uid=uid, report_name=f"R{uid}", report_fields_set=s,
             field_extraction_status="Matched" if s else "No fields found",
             total_risk_score=risk, last_run_date=TODAY - pd.Timedelta(days=10),
             times_run=10, owner="o", ownership_flags=[], shared="Yes",
             areas_used="a", landing_page="l", description="d", category="c",
             report_tag="t", data_source="ds", report_prompts="p",
             is_hard_rule=False, recommendation="Keep")
    r.update(kw)
    return r


def test_keeper_prefers_field_coverage(cfg):
    rich = rec(0, "a,b,c,d,e", risk=40)   # most fields but higher risk
    lean = rec(1, "a,b,c", risk=5)        # less risk but fewer fields
    keeper_uid, reason = choose_suggested_keeper([rich, lean])
    assert keeper_uid == 0
    assert "most report fields" in reason


def test_keeper_breaks_field_tie_on_lower_risk(cfg):
    a = rec(0, "a,b,c,d", risk=30)
    b = rec(1, "a,b,c,d", risk=8)         # same coverage, lower risk wins
    keeper_uid, _ = choose_suggested_keeper([a, b])
    assert keeper_uid == 1


def test_contained_member_is_delete_after_migration(cfg):
    keeper = rec(0, "a,b,c,d,e", risk=20)
    contained = rec(1, "a,b,c", risk=15)          # fully inside keeper
    by_uid = {0: keeper, 1: contained}
    g = DupGroup("DUP-0001", [0, 1])
    apply_duplicate_analysis(by_uid, [g], cfg)
    assert g.keeper_uid == 0
    assert keeper["is_suggested_keeper"] is True
    assert contained["field_containment_percent"] == 100.0
    assert contained["duplicate_classification"] == "One Report Contained In Another"
    assert resolve_recommendation(contained, g, by_uid, cfg) == "Delete Duplicate After Migration"
    assert resolve_recommendation(keeper, g, by_uid, cfg) == "Keep"


def test_strong_consolidation_member_action(cfg):
    keeper = rec(0, "a,b,c,d,e")
    cand = rec(1, "a,b,c,d,z")                     # 4/5 shared -> containment 80
    by_uid = {0: keeper, 1: cand}
    g = DupGroup("DUP-0001", [0, 1])
    apply_duplicate_analysis(by_uid, [g], cfg)
    assert cand["duplicate_classification"] == "Strong Consolidation Candidate"
    assert resolve_recommendation(cand, g, by_uid, cfg) == "Consolidate After Review"


def test_member_dup_columns_populated(cfg):
    keeper = rec(0, "a,b,c,d,e")
    member = rec(1, "a,b,c")
    by_uid = {0: keeper, 1: member}
    g = DupGroup("DUP-0001", [0, 1])
    apply_duplicate_analysis(by_uid, [g], cfg)
    assert member["suggested_keeper_report_name"] == "R0"
    assert member["field_count"] == 3
    assert member["shared_field_count"] == 3
    assert member["missing_fields_compared_to_keeper"] == []
    assert set(member["extra_fields_in_keeper"]) == {"d", "e"}
    assert member["consolidation_reason"]


def test_hard_rule_overrides(cfg):
    r = rec(0, is_hard_rule=True)
    assert resolve_recommendation(r, None, {0: r}, cfg) == "Delete - Hard Rule Match"


def test_non_duplicate_action_is_band(cfg):
    r = rec(0, recommendation="Low Priority Review")
    assert resolve_recommendation(r, None, {0: r}, cfg) == "Low Priority Review"
