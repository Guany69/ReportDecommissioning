"""Overall Decommissioning Score (0..100) + business-protection credits."""
import pandas as pd

from report_cleanup.overall import calculate_overall_score
from report_cleanup.protection import calculate_protection_credit

TODAY = pd.Timestamp("2026-06-05")


def test_spec_example_45_over_60_minus_12_is_63(cfg):
    res = calculate_overall_score(cleanup_points=45, protection_credit=12,
                                  hard_rule_triggered=False, cfg=cfg)
    assert res.cleanup_percentage == 75
    assert res.overall_score == 63


def test_protection_lowers_score(cfg):
    no_credit = calculate_overall_score(45, 0, False, cfg).overall_score
    with_credit = calculate_overall_score(45, 12, False, cfg).overall_score
    assert with_credit < no_credit


def test_normal_score_capped_at_99(cfg):
    res = calculate_overall_score(60, 0, False, cfg)  # 100% cleanup, no credit
    assert res.overall_score == 99  # reserved 100 for hard rules


def test_normal_score_floor_at_0(cfg):
    res = calculate_overall_score(0, 20, False, cfg)  # huge credit, no risk
    assert res.overall_score == 0


def test_hard_rule_is_always_100(cfg):
    res = calculate_overall_score(0, 0, True, cfg)
    assert res.overall_score == 100


def test_hard_rule_not_reduced_by_protection(cfg):
    res = calculate_overall_score(30, 20, True, cfg)
    assert res.overall_score == 100
    assert res.business_protection_credit == 0


# ---- protection credit tiers ----------------------------------------------
def _report(**kw):
    r = dict(distinct_requesters=0, areas_used="", worklet="No",
             landing_page="", report_landing_page="", shared="No")
    r.update(kw)
    return r


def test_recurrence_one_tier_only(cfg):
    strong, _ = calculate_protection_credit(_report(), {"classification": "Strong"}, cfg)
    moderate, _ = calculate_protection_credit(_report(), {"classification": "Moderate"}, cfg)
    assert strong == 8 and moderate == 4


def test_requester_breadth_tiers(cfg):
    five, _ = calculate_protection_credit(_report(distinct_requesters=5), None, cfg)
    few, _ = calculate_protection_credit(_report(distinct_requesters=3), None, cfg)
    assert five == 4 and few == 2


def test_context_credits_stack_to_max(cfg):
    credit, reasons = calculate_protection_credit(
        _report(distinct_requesters=9, areas_used="HR; Finance; Ops",
                worklet="Yes", shared="Yes"),
        {"classification": "Strong"}, cfg)
    # 8 (recur) + 4 (requesters) + 3 (areas) + 3 (placement) + 2 (shared) = 20 == cap
    assert credit == 20
    labels = [r.label for r in reasons]
    assert any("multiple distinct Areas" in l for l in labels)
    assert any("Shared = Yes" in l for l in labels)


def test_protection_credit_respects_lower_cap(cfg):
    import copy

    from report_cleanup.config import Config
    data = copy.deepcopy(cfg._data)
    data["business_protection"]["max_credit"] = 10
    capped_cfg = Config(data)
    credit, reasons = calculate_protection_credit(
        _report(distinct_requesters=9, areas_used="HR; Finance; Ops",
                worklet="Yes", shared="Yes"),
        {"classification": "Strong"}, capped_cfg)
    assert credit == 10
    assert any("capped" in r.label for r in reasons)


def test_insufficient_history_gives_no_recurrence_credit(cfg):
    credit, _ = calculate_protection_credit(_report(), {"classification": "Insufficient History"}, cfg)
    assert credit == 0
