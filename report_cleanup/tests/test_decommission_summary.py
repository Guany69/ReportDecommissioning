"""The focused decommission summary export: exactly four columns, score-ordered."""
import pandas as pd

from report_cleanup.export_excel import (
    SUMMARY_COLUMNS,
    decommission_summary_row,
    export_decommission_summary,
)
from report_cleanup.soft_scoring import Reason


def _record(uid, name, score, recommendation, reasons=None):
    return dict(
        report_uid=uid, report_name=name, overall_score=score,
        recommendation=recommendation, all_reasons=reasons or [],
    )


def test_summary_row_has_exactly_the_four_fields():
    r = _record(0, "Annual Headcount", 80, "Decommissioning Review",
                [Reason("usage", "No runs in 12 months", 30),
                 Reason("duplicate", "Likely Duplicate (Name Match)", None)])
    row = decommission_summary_row(r)
    assert list(row.keys()) == SUMMARY_COLUMNS
    assert row["Report Name"] == "Annual Headcount"
    assert row["Overall Score"] == 80
    assert row["Recommendation"] == "Decommissioning Review"
    # Reason trail joins points-bearing and point-less reasons.
    assert row["Reason Trail"] == "+30 No runs in 12 months; Likely Duplicate (Name Match)"


def test_export_writes_four_columns_ordered_by_score_desc(tmp_path):
    records = [
        _record(0, "Keep Me", 5, "Keep"),
        _record(1, "High Risk", 92, "High-Priority Decommissioning Review"),
        _record(2, "Middle", 60, "Owner Review / Monitor"),
    ]
    path = export_decommission_summary(records, tmp_path)
    assert path.exists() and path.suffix == ".xlsx"

    df = pd.read_excel(path, sheet_name="Decommission Summary", dtype=str)
    assert list(df.columns) == SUMMARY_COLUMNS
    # Strongest decommission candidate first, lowest last.
    assert list(df["Report Name"]) == ["High Risk", "Middle", "Keep Me"]


def test_export_handles_empty_records(tmp_path):
    """No reports -> a header-only sheet with the four columns, not a crash."""
    path = export_decommission_summary([], tmp_path)
    df = pd.read_excel(path, sheet_name="Decommission Summary", dtype=str)
    assert list(df.columns) == SUMMARY_COLUMNS
    assert len(df) == 0


def test_missing_score_sorts_last_and_renders_blank(tmp_path):
    records = [
        _record(0, "No Score", None, "Keep"),
        _record(1, "Scored", 40, "Owner Review / Monitor"),
    ]
    path = export_decommission_summary(records, tmp_path)
    df = pd.read_excel(path, sheet_name="Decommission Summary", dtype=str)
    assert list(df["Report Name"]) == ["Scored", "No Score"]
