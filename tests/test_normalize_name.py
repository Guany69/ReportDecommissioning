"""Exact (non-fuzzy) report-name normalization and the Comprehensive<->Runs join."""
import pandas as pd

from report_cleanup.clean import normalize_report_name
from report_cleanup.exec_rollup import build_exec_rollup
from report_cleanup.join import join_reports
from report_cleanup.validate import validate


def test_casefold_and_whitespace_collapse():
    assert normalize_report_name("  Active Employee Report ") == "active employee report"
    assert normalize_report_name("ACTIVE   employee\tReport") == "active employee report"


def test_none_and_nan_are_blank():
    assert normalize_report_name(None) == ""
    assert normalize_report_name(pd.NaT) == ""
    assert normalize_report_name(float("nan")) == ""


def test_join_is_case_insensitive_and_whitespace_trimmed():
    t1 = pd.DataFrame({"report_name": ["Active Employee Report"], "times_run": [1]})
    t2 = pd.DataFrame({"t2_report_name": ["  active employee report ", "ACTIVE EMPLOYEE REPORT"]})
    rollup = build_exec_rollup(t2)
    records, diag = join_reports(t1, rollup)
    assert diag["matched"] == 1
    assert records[0]["runs_exec_count"] == 2


def test_join_is_not_fuzzy():
    """De-noised variants ('Copy', 'v2') must NOT collapse onto the base name."""
    t1 = pd.DataFrame({"report_name": ["Headcount Summary"], "times_run": [1]})
    t2 = pd.DataFrame({"t2_report_name": ["Headcount Summary (Copy)", "Headcount Summary v2"]})
    rollup = build_exec_rollup(t2)
    records, diag = join_reports(t1, rollup)
    assert diag["matched"] == 0
    assert records[0]["runs_exec_count"] == 0


def test_duplicate_normalized_comprehensive_names_flagged():
    t1 = pd.DataFrame({"Custom Report": ["Report A", "report a", "Unique B"]})
    vr = validate(t1, None, {"report_name": "Custom Report"}, {})
    assert vr.duplicate_report_names == ["report a"]
    assert any("duplicate normalized" in w.lower() for w in vr.warnings)
    assert not vr.has_fatal()  # surfaced as a data-quality warning, not a merge
