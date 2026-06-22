"""The exported row must carry the new risk, flag, and duplicate columns."""
import pandas as pd

from report_cleanup.export_excel import report_row

RISK_COLS = ["usage_risk", "age_risk", "usage_context_risk", "total_risk_score", "recommendation"]
FLAG_COLS = ["ownership_flags", "data_quality_flags", "all_flags"]
DUP_COLS = [
    "duplicate_group_id", "is_suggested_keeper", "suggested_keeper_report_name",
    "candidate_similarity_score", "report_name_similarity_percent",
    "field_count", "shared_field_count", "field_similarity_percent",
    "field_containment_percent", "missing_fields_compared_to_keeper",
    "extra_fields_in_keeper", "duplicate_classification", "suggested_action",
    "consolidation_reason",
]


def _record():
    return dict(
        report_uid=0, report_name="R", times_run=5, last_run_date=pd.NaT,
        usage_risk=10, age_risk=5, usage_context_risk=4, total_risk_score=19,
        recommendation="Low Priority Review",
        ownership_flags=["Missing Owner"], data_quality_flags=["Missing Description"],
        all_flags=["Missing Owner", "Missing Description"],
        dup_group_id="DUP-0001", is_suggested_keeper=False,
        suggested_keeper_report_name="Keeper", candidate_similarity_score=88.0,
        report_name_similarity_percent=90.0, field_count=3, shared_field_count=3,
        field_similarity_percent=60.0, field_containment_percent=100.0,
        missing_fields_compared_to_keeper=[], extra_fields_in_keeper=["x"],
        duplicate_classification="Strong Consolidation Candidate",
        suggested_action="Consolidate After Review",
        consolidation_reason="because reasons", all_reasons=[],
    )


def test_output_has_all_new_columns():
    row = report_row(_record())
    for col in RISK_COLS + FLAG_COLS + DUP_COLS:
        assert col in row, f"missing output column: {col}"


def test_flags_joined_with_semicolons():
    row = report_row(_record())
    assert row["all_flags"] == "Missing Owner; Missing Description"


NEW_HEADLINE_COLS = [
    "Overall Score", "Recommendation", "Hard Rule Triggered", "Hard Rule Name",
    "Effective Last Run Date", "Comprehensive Exec Count", "Runs Exec Count (6mo window)",
    "Cleanup Risk Points", "Cleanup %", "Business Protection Credit",
    "Cleanup Reasons", "Protection Reasons",
    "Recurrence", "Recurrence Cadence", "Distinct Requesters",
    "Potential Duplicate", "Potential Duplicate Of", "Duplicate Similarity",
    "Duplicate Relationship", "Field Jaccard %", "Smaller Report Containment %",
]


def test_output_has_new_headline_columns():
    row = report_row(_record())
    for col in NEW_HEADLINE_COLS:
        assert col in row, f"missing new output column: {col}"


def test_overall_score_and_recurrence_render():
    rec = _record()
    rec.update(overall_score=63, cleanup_risk_points=45, cleanup_percentage=75,
               business_protection_credit=12, recurrence_classification="Strong",
               recurrence_cadence="monthly", potential_duplicate=True,
               potential_duplicate_of="Other", duplicate_similarity=91.0,
               duplicate_relationship="Nearly Identical")
    row = report_row(rec)
    assert row["Overall Score"] == 63
    assert row["Recurrence"] == "Strong"
    assert row["Potential Duplicate"] == "Yes"
    assert row["Potential Duplicate Of"] == "Other"
