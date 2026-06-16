"""Two-stage duplicate detection: Stage-1 gate then Stage-2 field comparison.

detect_duplicates now returns (groups, meta_only_pairs).
report_fields_set is set directly on records (no longer parsed from metadata text).
"""
import pandas as pd

from report_cleanup.dedup import detect_duplicates, should_compare_report_fields


def rec(uid, name, fields="", **kw):
    # Build report_fields_set directly — detect_duplicates no longer parses text.
    field_set = {f.strip() for f in fields.split(",") if f.strip()} if fields else set()
    r = dict(report_uid=uid, report_name=name,
             report_fields_set=field_set,
             field_extraction_status="Matched" if field_set else "No fields found",
             data_source="DS1", category="Cat1", report_tag="Tag1",
             report_type="Advanced", worklet="No",
             last_run_date=pd.NaT, times_run=0, last_updated=pd.NaT,
             owner="o", total_risk_score=10)
    r.update(kw)
    return r


# ---- Stage 1: candidate gate ----------------------------------------------
def test_not_compared_when_dissimilar(cfg):
    a = rec(0, "Termination Report", data_source="Terminations", category="HR", report_tag="T")
    b = rec(1, "Terminations Monthly", data_source="Payroll", category="Finance", report_tag="F")
    assert should_compare_report_fields(a, b, cfg) is False


def test_compared_when_name_similarity_strong(cfg):
    a = rec(0, "Active Worker List", data_source="A", category="X", report_tag="P")
    b = rec(1, "Active Worker Listing", data_source="B", category="Y", report_tag="Q")
    assert should_compare_report_fields(a, b, cfg) is True


def test_moderate_name_plus_data_source_alone_not_compared(cfg):
    # Tightened: data source is only a SUPPORTING signal now. Moderate name
    # similarity + a shared data source (but differing category/tag and no shared
    # fields) is no longer enough on its own to trigger a field comparison.
    a = rec(0, "Worker Roster", data_source="Workers", category="X", report_tag="P")
    b = rec(1, "Worker Headcount Roster", data_source="Workers", category="Y", report_tag="Q")
    assert should_compare_report_fields(a, b, cfg) is False


def test_compared_when_moderate_name_plus_category_and_tag(cfg):
    a = rec(0, "Worker Roster", data_source="Workers", category="HR", report_tag="Roster")
    b = rec(1, "Worker Headcount Roster", data_source="Payroll", category="HR", report_tag="Roster")
    assert should_compare_report_fields(a, b, cfg) is True


# ---- Stage 2: grouping + classification -----------------------------------
def test_contained_report_is_consolidation_candidate(cfg):
    keeper = rec(0, "Worker Headcount Roster",
                 fields="Employee ID, Employee Name, Manager, Location, Hire Date")
    smaller = rec(1, "Worker Headcount Roster Detail",
                  fields="Employee ID, Employee Name, Manager")
    groups, _ = detect_duplicates([keeper, smaller], cfg)
    assert len(groups) == 1
    assert set(groups[0].members) == {0, 1}


def test_distinct_reports_not_grouped(cfg):
    a = rec(0, "Headcount Snapshot", data_source="Employees", category="Worker", report_tag="HC",
            fields="Employee ID, Location")
    b = rec(1, "Terminations Monthly", data_source="Terminations", category="Exits", report_tag="Term",
            fields="Termination Date, Reason")
    groups, _ = detect_duplicates([a, b], cfg)
    assert groups == []


def test_near_identical_fields_group(cfg):
    a = rec(0, "Annual Performance", fields="Employee ID, Manager, Rating, Period")
    b = rec(1, "Annual Performance Review", fields="Employee ID, Manager, Rating, Period")
    groups, _ = detect_duplicates([a, b], cfg)
    assert len(groups) == 1 and set(groups[0].members) == {0, 1}


# ---- Metadata-only pairs (no field sets) ----------------------------------
def test_meta_only_pair_when_fields_missing(cfg):
    """Stage-1 passes but both records have empty field sets -> meta_only_pairs, not a group."""
    a = rec(0, "Active Worker List", fields="")   # no field set
    b = rec(1, "Active Worker Listing", fields="") # no field set
    groups, meta_only = detect_duplicates([a, b], cfg)
    assert groups == []
    assert (0, 1) in meta_only or (1, 0) in meta_only


def test_one_sided_field_set_goes_to_meta_only(cfg):
    """One report has fields, the other doesn't -> cannot confirm via fields."""
    a = rec(0, "Active Worker List", fields="Employee ID, Name, Location")
    b = rec(1, "Active Worker Listing", fields="")
    groups, meta_only = detect_duplicates([a, b], cfg)
    assert groups == []
    assert len(meta_only) == 1
