"""Ownership & data-quality flags — advisory, with no effect on the risk score."""
import pandas as pd

from report_cleanup.flags import (build_all_flags, generate_data_quality_flags,
                                  generate_ownership_flags)
from report_cleanup.soft_scoring import score_report

TODAY = pd.Timestamp("2026-06-05")


def base(**kw):
    r = dict(report_name="R", last_run_date=TODAY, times_run=50, last_updated=TODAY,
             areas_used="x", landing_page="x", shared="Yes",
             owner="o", t2_owner="o", created_by="c",
             description="d", category="c", report_tag="t", data_source="s")
    r.update(kw)
    return r


def test_missing_owner_flag(cfg):
    assert "Missing Owner" in generate_ownership_flags(base(owner=""), cfg)


def test_owner_mismatch_flag(cfg):
    assert "Owner Mismatch" in generate_ownership_flags(base(owner="A", t2_owner="B"), cfg)


def test_owner_match_no_flag(cfg):
    assert "Owner Mismatch" not in generate_ownership_flags(base(owner="A", t2_owner="a"), cfg)


def test_missing_creator_flag(cfg):
    assert "Missing Creator" in generate_ownership_flags(base(created_by=""), cfg)


def test_missing_description_flag(cfg):
    assert "Missing Description" in generate_data_quality_flags(base(description=""), cfg)


def test_data_quality_flags_complete(cfg):
    flags = generate_data_quality_flags(
        base(description="", category="", report_tag="", data_source=""), cfg)
    assert set(flags) == {"Missing Description", "Missing Category",
                          "Missing Report Tag", "Missing Data Source"}


def test_inactive_owner_inert_without_status_column(cfg):
    # No owner_active/owner_status column -> flag stays empty.
    assert "Inactive Owner" not in generate_ownership_flags(base(owner="J. Doe"), cfg)


def test_inactive_owner_from_status_column(cfg):
    assert "Inactive Owner" in generate_ownership_flags(base(owner_active="No"), cfg)


def test_build_all_flags_combines(cfg):
    out = build_all_flags(base(owner="", description=""), cfg)
    assert "Missing Owner" in out["ownership_flags"]
    assert "Missing Description" in out["data_quality_flags"]
    assert set(out["all_flags"]) == set(out["ownership_flags"] + out["data_quality_flags"])


def test_flags_do_not_change_score(cfg):
    clean = score_report(base(), cfg, today=TODAY).total_risk_score
    flagged = score_report(base(owner="", created_by="", description="", category="",
                                report_tag="", data_source=""), cfg, today=TODAY).total_risk_score
    assert clean == flagged == 0
