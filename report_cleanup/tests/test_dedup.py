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
    """Moderate name similarity (>=strong gate, <name_match) + empty field sets ->
    meta_only_pairs, not a group."""
    a = rec(0, "Annual Performance", fields="")        # name sim ~84 -> not a name edge
    b = rec(1, "Annual Performance Review", fields="")
    groups, meta_only = detect_duplicates([a, b], cfg)
    assert groups == []
    assert (0, 1) in meta_only or (1, 0) in meta_only


def test_one_sided_field_set_goes_to_meta_only(cfg):
    """One report has fields, the other doesn't, names only moderately similar ->
    cannot confirm; goes to meta_only (not a name-based group)."""
    a = rec(0, "Annual Performance", fields="Employee ID, Name, Location")
    b = rec(1, "Annual Performance Review", fields="")
    groups, meta_only = detect_duplicates([a, b], cfg)
    assert groups == []
    assert len(meta_only) == 1


def test_name_match_groups_without_shared_fields(cfg):
    """A strong name match (>= name_match) forms a group even when field IDs don't
    overlap — the core fix for copies that live in their own Fields-export rows."""
    a = rec(0, "Active Worker List", fields="aa, bb, cc")
    b = rec(1, "Active Worker Listing", fields="xx, yy, zz")  # disjoint fields, name ~92
    groups, meta_only = detect_duplicates([a, b], cfg)
    assert len(groups) == 1 and set(groups[0].members) == {0, 1}
    assert meta_only == []


def test_copy_of_groups_with_original(cfg):
    """'Copy of X' groups with 'X' (de-noising strips 'copy'/'of') even with no
    shared field IDs."""
    a = rec(0, "Headcount Summary", fields="h1, h2")
    b = rec(1, "Copy of Headcount Summary", fields="z9")
    groups, _ = detect_duplicates([a, b], cfg)
    assert len(groups) == 1 and set(groups[0].members) == {0, 1}


def test_year_variants_not_grouped_by_name(cfg):
    """Year-variant reports must NOT name-match. The shared 'Annual Review' prefix
    makes them genuine candidates (blocking does NOT save us), so this exercises the
    real guard: the differing numeric tokens reject the match even though the char
    ratio is >90."""
    a = rec(0, "Annual Review 2017", fields="a1, a2")
    b = rec(1, "Annual Review 2018", fields="b1, b2")  # disjoint fields, only the year differs
    groups, _ = detect_duplicates([a, b], cfg)
    assert groups == []


def test_quarter_variants_not_grouped_by_name(cfg):
    """Fiscal/quarter numerics differing also blocks a name match (FY17 vs FY18)."""
    a = rec(0, "FY17 Headcount Detail", fields="a1, a2")
    b = rec(1, "FY18 Headcount Detail", fields="b1, b2")
    groups, _ = detect_duplicates([a, b], cfg)
    assert groups == []


def test_blank_and_noise_only_names_never_name_match(cfg):
    """Empty, missing, and noise-only names de-noise to "" (RapidFuzz scores two
    empty strings 100). They must NEVER form a name-based group, even when they
    share a field ID — otherwise every nameless report collapses into one cluster."""
    # One shared field id makes the pair a candidate, but balanced 5-field sets keep
    # Jaccard (~11%) and containment (20%) well below an edge — so the ONLY thing
    # that could group them is a name match, which must not fire on empty names.
    for na, nb in [("", ""), (None, None), ("Copy", "Copy"), ("Report", "Report")]:
        a = rec(0, na, fields="f1, f2, f3, f4, f5")
        b = rec(1, nb, fields="f1, g2, g3, g4, g5")
        groups, _ = detect_duplicates([a, b], cfg)
        assert groups == [], f"{na!r} vs {nb!r} should not group on name"


def test_same_year_copy_still_groups(cfg):
    """The digit guard must not break legitimate copies: a copy keeps the original's
    year, so identical numeric tokens still allow the match."""
    a = rec(0, "2017 Year End Summary", fields="a1, a2")
    b = rec(1, "Copy of 2017 Year End Summary", fields="z9")  # disjoint fields, same year
    groups, _ = detect_duplicates([a, b], cfg)
    assert len(groups) == 1 and set(groups[0].members) == {0, 1}
