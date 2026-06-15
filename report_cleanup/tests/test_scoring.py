"""Risk scoring (0..60). `today` is injected so date tiers are deterministic."""
import pandas as pd

from report_cleanup.soft_scoring import (calculate_age_risk,
                                         calculate_total_risk_score,
                                         calculate_usage_context_risk,
                                         calculate_usage_risk, score_report)

TODAY = pd.Timestamp("2026-06-05")


def base(**kw):
    """A clean, low-risk report; override fields per test."""
    r = dict(report_name="R",
             last_run_date=TODAY, times_run=50,
             last_updated=TODAY, areas_used="x", landing_page="x", shared="Yes",
             owner="o", t2_owner="o", created_by="c",
             description="d", category="c", report_tag="t", data_source="s")
    r.update(kw)
    return r


def usage(r, cfg):
    return calculate_usage_risk(r, TODAY, cfg)[0]


def age(r, cfg):
    return calculate_age_risk(r, TODAY, cfg)[0]


def context(r, cfg):
    return calculate_usage_context_risk(r, cfg)[0]


# ---- Last Run Date tiers --------------------------------------------------
def test_last_run_null_gives_25(cfg):
    assert usage(base(last_run_date=pd.NaT, times_run=50), cfg) == 25


def test_last_run_over_3y_gives_25(cfg):
    assert usage(base(last_run_date=TODAY - pd.Timedelta(days=1200), times_run=50), cfg) == 25


def test_last_run_over_2y_gives_20(cfg):
    assert usage(base(last_run_date=TODAY - pd.Timedelta(days=800), times_run=50), cfg) == 20


def test_last_run_over_1y_gives_10(cfg):
    assert usage(base(last_run_date=TODAY - pd.Timedelta(days=400), times_run=50), cfg) == 10


# ---- Number of Times Run tiers --------------------------------------------
def test_times_run_zero_gives_10(cfg):
    assert usage(base(times_run=0), cfg) == 10


def test_times_run_1_3_gives_7(cfg):
    assert usage(base(times_run=2), cfg) == 7


def test_times_run_4_10_gives_3(cfg):
    assert usage(base(times_run=8), cfg) == 3


def test_usage_cap(cfg):
    r = base(last_run_date=pd.NaT, times_run=0)   # 25 + 10 = 35
    assert usage(r, cfg) == 35
    assert usage(r, cfg) <= cfg.scoring["usage_cap"]


# ---- Age (Last Updated) tiers ---------------------------------------------
def test_last_updated_null_gives_5(cfg):
    assert age(base(last_updated=pd.NaT), cfg) == 5


def test_last_updated_over_3y_gives_10(cfg):
    assert age(base(last_updated=TODAY - pd.Timedelta(days=1200)), cfg) == 10


def test_last_updated_over_2y_gives_7(cfg):
    assert age(base(last_updated=TODAY - pd.Timedelta(days=800)), cfg) == 7


# ---- Usage context (stacking) ---------------------------------------------
def test_areas_blank_gives_6(cfg):
    assert context(base(areas_used=""), cfg) == 6


def test_landing_blank_gives_5(cfg):
    assert context(base(landing_page=""), cfg) == 5


def test_shared_no_gives_4(cfg):
    assert context(base(shared="No"), cfg) == 4
    assert context(base(shared="false"), cfg) == 4
    assert context(base(shared="N"), cfg) == 4


# ---- Flags do not move the score ------------------------------------------
def test_ownership_does_not_change_score(cfg):
    clean = score_report(base(), cfg, today=TODAY).total_risk_score
    no_owner = score_report(base(owner="", created_by=""), cfg, today=TODAY).total_risk_score
    assert clean == no_owner


def test_data_quality_does_not_change_score(cfg):
    clean = score_report(base(), cfg, today=TODAY).total_risk_score
    no_meta = score_report(base(description="", category="", report_tag="", data_source=""),
                           cfg, today=TODAY).total_risk_score
    assert clean == no_meta


# ---- Totals + bands -------------------------------------------------------
def test_total_and_band(cfg):
    res = calculate_total_risk_score(base(last_run_date=pd.NaT, times_run=0,
                                          last_updated=pd.NaT, areas_used="",
                                          landing_page="", shared="No"),
                                     TODAY, cfg)
    # usage 35 + age 5 + context 15 = 55 -> Decommission Review
    assert res.total_risk_score == 55
    assert res.recommendation == "Decommission Review"
    assert 0 <= res.total_risk_score <= 60


def test_clean_report_keeps(cfg):
    res = score_report(base(), cfg, today=TODAY)
    assert res.total_risk_score == 0
    assert res.recommendation == "Keep"
