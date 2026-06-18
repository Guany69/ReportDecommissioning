"""The focused decommission summary export: the four content columns plus the two
duplicate-identification columns (Duplicate Status, Duplicate Group ID), with rows
in grouped order (duplicate groups first as contiguous blocks, then non-duplicates).
"""
import pandas as pd

from report_cleanup.export_excel import (
    SUMMARY_COLUMNS,
    assign_duplicate_group_ids,
    decommission_summary_row,
    duplicate_identification,
    export_decommission_summary,
)
from report_cleanup.soft_scoring import Reason


def _record(uid, name, score=10, recommendation="Keep", reasons=None, dup_group_id=None):
    """Mimic a record as produced by the (existing) pipeline. dup_group_id is the
    internal group id stamped by recommend.apply_duplicate_analysis onto every member
    of a detected duplicate group; absent for non-duplicates."""
    r = dict(report_uid=uid, report_name=name, overall_score=score,
             recommendation=recommendation, all_reasons=reasons or [])
    if dup_group_id:
        r["dup_group_id"] = dup_group_id
    return r


def _read(path):
    # keep_default_na=False so a blank cell stays "" (not NaN) — lets us assert that
    # non-duplicate group ids are truly blank in the exported file.
    return pd.read_excel(path, sheet_name="Decommission Summary", dtype=str,
                         keep_default_na=False)


# ---- columns / row content -------------------------------------------------
def test_summary_row_has_the_six_columns():
    r = _record(0, "Annual Headcount", 80, "Decommissioning Review",
                [Reason("usage", "No runs in 12 months", 30),
                 Reason("duplicate", "Likely Duplicate (Name Match)", None)])
    row = decommission_summary_row(r, "Duplicate", "DG-001")
    assert list(row.keys()) == SUMMARY_COLUMNS
    assert row["Report Name"] == "Annual Headcount"
    assert row["Overall Score"] == 80
    assert row["Recommendation"] == "Decommissioning Review"
    assert row["Reason Trail"] == "+30 No runs in 12 months; Likely Duplicate (Name Match)"
    assert row["Duplicate Status"] == "Duplicate"
    assert row["Duplicate Group ID"] == "DG-001"


def test_recommendation_score_and_reason_trail_pass_through_unchanged():
    """Adding the duplicate columns must not disturb the existing content columns."""
    r = _record(1, "X", 63, "Owner Review / Monitor",
                [Reason("usage", "Run monthly", -15)])
    row = decommission_summary_row(r)
    assert row["Overall Score"] == 63
    assert row["Recommendation"] == "Owner Review / Monitor"
    assert row["Reason Trail"] == "-15 Run monthly"


# ---- group id assignment ---------------------------------------------------
def test_two_matched_reports_share_one_group_id():
    recs = [_record(0, "Apple", dup_group_id="DUP-0001"),
            _record(1, "Avocado", dup_group_id="DUP-0001")]
    dg = assign_duplicate_group_ids(recs)
    assert dg == {"DUP-0001": "DG-001"}


def test_separate_groups_get_different_ids_ordered_by_first_name():
    # Group with the alphabetically-earliest first member name becomes DG-001.
    recs = [_record(0, "Cherry", dup_group_id="DUP-0002"),
            _record(1, "Cranberry", dup_group_id="DUP-0002"),
            _record(2, "Apple", dup_group_id="DUP-0001"),
            _record(3, "Avocado", dup_group_id="DUP-0001")]
    dg = assign_duplicate_group_ids(recs)
    assert dg == {"DUP-0001": "DG-001", "DUP-0002": "DG-002"}


def test_transitive_matches_form_one_group():
    """A-B and B-C already collapse to one internal group upstream (union-find), so
    all three carry the same dup_group_id and must get the same DG id."""
    recs = [_record(0, "A", dup_group_id="DUP-0005"),
            _record(1, "B", dup_group_id="DUP-0005"),
            _record(2, "C", dup_group_id="DUP-0005")]
    dg = assign_duplicate_group_ids(recs)
    assert dg == {"DUP-0005": "DG-001"}


def test_nonduplicates_excluded_from_id_assignment():
    recs = [_record(0, "Lonely")]  # no dup_group_id
    assert assign_duplicate_group_ids(recs) == {}


def test_duplicate_identification_shared_helper():
    """The helper the app and the export both consume: per-uid status + DG id."""
    recs = [_record(0, "Bravo", dup_group_id="DUP-0001"),
            _record(1, "Alpha", dup_group_id="DUP-0001"),
            _record(2, "Solo")]
    ident = duplicate_identification(recs)
    assert ident[0][0] == "Duplicate" and ident[0][1] == "DG-001"
    assert ident[1][0] == "Duplicate" and ident[1][1] == "DG-001"
    assert ident[2][0] == "Not a Duplicate" and ident[2][1] == ""
    # Non-duplicate sorts after duplicates (primary key 1 vs 0).
    assert ident[2][2][0] == 1 and ident[0][2][0] == 0


# ---- exported file: columns, blanks, ordering ------------------------------
def test_nonduplicate_group_id_is_blank_not_none(tmp_path):
    recs = [_record(0, "Solo", recommendation="Keep")]
    df = _read(export_decommission_summary(recs, tmp_path))
    assert list(df.columns) == SUMMARY_COLUMNS
    assert df.loc[0, "Duplicate Status"] == "Not a Duplicate"
    assert df.loc[0, "Duplicate Group ID"] == ""          # blank, not "None"/"nan"


def test_grouped_ordering_blocks_then_nonduplicates(tmp_path):
    recs = [
        _record(0, "Echo", dup_group_id="DUP-0002"),      # group 2
        _record(1, "Foxtrot"),                            # non-dup
        _record(2, "Bravo", dup_group_id="DUP-0001"),     # group 1
        _record(3, "Alpha", dup_group_id="DUP-0001"),     # group 1
        _record(4, "Delta", dup_group_id="DUP-0002"),     # group 2
        _record(5, "Golf"),                               # non-dup
    ]
    df = _read(export_decommission_summary(recs, tmp_path))

    names = list(df["Report Name"])
    statuses = list(df["Duplicate Status"])
    gids = list(df["Duplicate Group ID"])

    # Group 1 (first names Alpha/Bravo) before group 2 (Delta/Echo); within a group
    # sorted by name; non-duplicates (Foxtrot, Golf) last, sorted by name.
    assert names == ["Alpha", "Bravo", "Delta", "Echo", "Foxtrot", "Golf"]
    assert gids == ["DG-001", "DG-001", "DG-002", "DG-002", "", ""]
    assert statuses == ["Duplicate", "Duplicate", "Duplicate", "Duplicate",
                        "Not a Duplicate", "Not a Duplicate"]

    # Every group's members are contiguous (no other group/non-dup interleaved).
    for gid in ("DG-001", "DG-002"):
        idx = [i for i, g in enumerate(gids) if g == gid]
        assert idx == list(range(idx[0], idx[-1] + 1)), f"{gid} members not adjacent"


def test_empty_records_header_only(tmp_path):
    df = _read(export_decommission_summary([], tmp_path))
    assert list(df.columns) == SUMMARY_COLUMNS
    assert len(df) == 0


def test_ordering_and_ids_are_deterministic_across_runs(tmp_path):
    recs = [
        _record(0, "Zulu", dup_group_id="DUP-0009"),
        _record(1, "Mike", dup_group_id="DUP-0003"),
        _record(2, "Alpha", dup_group_id="DUP-0009"),
        _record(3, "November"),
        _record(4, "Bravo", dup_group_id="DUP-0003"),
    ]
    d1 = _read(export_decommission_summary(recs, tmp_path / "run1"))
    d2 = _read(export_decommission_summary(recs, tmp_path / "run2"))
    assert list(d1["Report Name"]) == list(d2["Report Name"])
    assert list(d1["Duplicate Group ID"]) == list(d2["Duplicate Group ID"])
