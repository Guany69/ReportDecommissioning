"""Tests for the field_rollup module (table3 ingest, rollup, and attachment)."""
from __future__ import annotations

import pandas as pd
import pytest

from report_cleanup.dedup import DupGroup, detect_duplicates
from report_cleanup.field_rollup import (
    FIELD_EXPORT_MODE_CATALOG_ONLY,
    FIELD_EXPORT_MODE_MISSING,
    FIELD_EXPORT_MODE_REPORT_MAPPING,
    STATUS_CATALOG_ONLY,
    STATUS_MATCHED,
    STATUS_NOT_PROVIDED,
    STATUS_NO_FIELDS,
    attach_report_fields,
    build_report_field_rollup,
    parse_where_used,
    validate_field_table,
)
from report_cleanup.recommend import choose_suggested_keeper


# ---- Helpers ---------------------------------------------------------------

def make_t3(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical t3 DataFrame (already renamed to t3_* keys)."""
    return pd.DataFrame(rows)


def base_report(uid=0, name="Worker Profile", **kw):
    r = dict(report_uid=uid, report_name=name, report_fields_set=set(),
             field_extraction_status=STATUS_NO_FIELDS,
             total_risk_score=10, last_run_date=pd.NaT,
             times_run=5, owner="Alice", ownership_flags=[], shared="Yes",
             areas_used="HR", landing_page="Home", description="d", category="c",
             report_tag="t", data_source="ds", report_prompts="p",
             is_hard_rule=False, recommendation="Keep")
    r.update(kw)
    return r


# ---- validate_field_table --------------------------------------------------

def test_validate_missing_returns_missing_mode():
    vr = validate_field_table(None, {})
    assert vr.mode == FIELD_EXPORT_MODE_MISSING
    assert vr.warnings


def test_validate_empty_df_returns_missing_mode():
    vr = validate_field_table(pd.DataFrame(), {})
    assert vr.mode == FIELD_EXPORT_MODE_MISSING


def test_validate_with_report_name_mapped():
    t3 = make_t3([{"t3_report_name": "Worker Profile", "t3_field_name": "Employee ID"}])
    t3_map = {"t3_report_name": "t3_report_name", "t3_field_name": "t3_field_name"}
    vr = validate_field_table(t3, t3_map)
    assert vr.mode == FIELD_EXPORT_MODE_REPORT_MAPPING
    assert not vr.errors


def test_validate_with_where_used_mapped():
    t3 = make_t3([{"t3_where_used": "Worker Profile", "t3_field_name": "Employee ID"}])
    t3_map = {"t3_where_used": "t3_where_used", "t3_field_name": "t3_field_name"}
    vr = validate_field_table(t3, t3_map)
    assert vr.mode == FIELD_EXPORT_MODE_REPORT_MAPPING


def test_validate_catalog_only_when_no_report_key():
    t3 = make_t3([{"t3_field_name": "Employee ID", "t3_business_object": "Worker",
                   "t3_domain": "Staffing"}])
    t3_map = {"t3_field_name": "t3_field_name", "t3_business_object": "t3_business_object"}
    vr = validate_field_table(t3, t3_map)
    assert vr.mode == FIELD_EXPORT_MODE_CATALOG_ONLY
    assert any("catalog" in w.lower() for w in vr.warnings)


# ---- parse_where_used (line breaks ONLY: \n, \r\n, \r) ---------------------

def test_parse_where_used_newline_lf():
    keys = parse_where_used("Worker Profile\nHeadcount Snapshot")
    assert keys == ["worker profile", "headcount snapshot"]


def test_parse_where_used_crlf():
    keys = parse_where_used("Worker Profile\r\nHeadcount Snapshot\r\nAnnual Review")
    assert len(keys) == 3


def test_parse_where_used_cr_only():
    keys = parse_where_used("Worker Profile\rHeadcount Snapshot")
    assert len(keys) == 2


def test_parse_where_used_blank_lines_dropped():
    keys = parse_where_used("Worker Profile\n\n   \nHeadcount Snapshot\n")
    assert keys == ["worker profile", "headcount snapshot"]


def test_parse_where_used_case_and_whitespace_normalized():
    keys = parse_where_used("  WORKER   profile \nworker profile")
    assert keys == ["worker profile"]  # deduped after exact normalization


def test_parse_where_used_separators_are_not_split():
    # Semicolons / pipes / commas can appear inside report names -> NOT separators.
    assert parse_where_used("Worker Profile; Annual Review") == ["worker profile; annual review"]
    assert parse_where_used("A | B, C") == ["a | b, c"]


def test_parse_where_used_blank_returns_empty():
    assert parse_where_used("") == []
    assert parse_where_used(None) == []


# ---- build_report_field_rollup ---------------------------------------------

def test_rollup_groups_by_report_name(cfg):
    t3 = make_t3([
        {"t3_report_name": "Worker Profile", "t3_field_name": "Employee ID",
         "t3_business_object": "Worker", "t3_related_business_object": "",
         "t3_report_field_type": "", "t3_domain": "HR", "t3_categories": "",
         "t3_authorized_usage": ""},
        {"t3_report_name": "Worker Profile", "t3_field_name": "Manager",
         "t3_business_object": "Worker", "t3_related_business_object": "",
         "t3_report_field_type": "", "t3_domain": "HR", "t3_categories": "",
         "t3_authorized_usage": ""},
        {"t3_report_name": "Headcount Snapshot", "t3_field_name": "Employee ID",
         "t3_business_object": "Worker", "t3_related_business_object": "",
         "t3_report_field_type": "", "t3_domain": "HR", "t3_categories": "",
         "t3_authorized_usage": ""},
    ])
    result = build_report_field_rollup(t3, cfg)
    rollup = result["rollup"]
    # "Worker Profile" -> normalized key should have 2 fields, "Headcount Snapshot" -> 1
    assert any(v["field_count"] == 2 for v in rollup.values())
    assert any(v["field_count"] == 1 for v in rollup.values())
    assert result["diagnostics"]["total_report_field_links"] == 3
    assert result["diagnostics"]["total_unmatched_rows"] == 0


def test_rollup_unmatched_when_no_report_key(cfg):
    t3 = make_t3([
        {"t3_report_name": "", "t3_field_name": "Employee ID",
         "t3_business_object": "Worker", "t3_related_business_object": "",
         "t3_report_field_type": "", "t3_domain": "", "t3_categories": "",
         "t3_authorized_usage": ""},
    ])
    result = build_report_field_rollup(t3, cfg)
    assert result["diagnostics"]["total_unmatched_rows"] == 1
    assert len(result["unmatched_rows"]) == 1


def test_rollup_where_used_expands_to_multiple_reports(cfg):
    t3 = make_t3([
        {"t3_report_name": "", "t3_where_used": "Worker Profile\nHeadcount Snapshot",
         "t3_field_name": "Employee ID", "t3_business_object": "Worker",
         "t3_related_business_object": "", "t3_report_field_type": "",
         "t3_domain": "", "t3_categories": "", "t3_authorized_usage": ""},
    ])
    result = build_report_field_rollup(t3, cfg)
    assert result["diagnostics"]["total_report_field_links"] == 2
    assert len(result["rollup"]) == 2


def test_same_field_row_maps_to_all_listed_reports(cfg):
    # One Fields row whose Where_Used lists three reports -> the SAME field ID is
    # added to all three report field sets.
    t3 = make_t3([
        {"t3_where_used": "Active Employee Report\nHeadcount Summary\nEmployee Audit",
         "t3_field_name": "Employee ID", "t3_business_object": "Worker"},
    ])
    result = build_report_field_rollup(t3, cfg)
    rollup = result["rollup"]
    assert set(rollup) == {"active employee report", "headcount summary", "employee audit"}
    field_ids = {next(iter(v["field_set"])) for v in rollup.values()}
    assert len(field_ids) == 1  # identical field ID shared across the three reports


def test_blank_where_used_tracked(cfg):
    t3 = make_t3([
        {"t3_where_used": "", "t3_field_name": "Orphan Field", "t3_business_object": "Worker"},
    ])
    result = build_report_field_rollup(t3, cfg)
    assert result["diagnostics"]["blank_where_used_rows"] == 1
    assert result["diagnostics"]["total_unmatched_rows"] == 1


def test_unmatched_where_used_names_recorded(cfg):
    t3 = make_t3([
        {"t3_where_used": "Known Report\nGhost Report", "t3_field_name": "F", "t3_business_object": "B"},
    ])
    result = build_report_field_rollup(t3, cfg)
    records = [base_report(0, "Known Report")]
    attach_report_fields(records, result, FIELD_EXPORT_MODE_REPORT_MAPPING, cfg)
    assert records[0]["field_extraction_status"] == STATUS_MATCHED
    assert result["unmatched_where_used_names"] == ["ghost report"]


# ---- attach_report_fields --------------------------------------------------

def test_attach_matched(cfg):
    rollup_result = {
        "rollup": {"worker profile": {
            "field_set": {"worker|employee id||"},
            "field_count": 1,
            "field_list_text": "Employee ID",
            "field_signature": "worker|employee id||",
            "business_objects_used": ["Worker"],
            "domains_used": ["HR"],
            "categories_used": [],
        }},
        "diagnostics": {}, "unmatched_rows": [], "individual_rows": [],
    }
    records = [base_report(0, "Worker Profile")]
    attach_report_fields(records, rollup_result, FIELD_EXPORT_MODE_REPORT_MAPPING, cfg)
    r = records[0]
    assert r["field_extraction_status"] == STATUS_MATCHED
    assert "worker|employee id||" in r["report_fields_set"]
    assert r["field_count"] == 1


def test_attach_no_match_gives_no_fields_status(cfg):
    rollup_result = {"rollup": {}, "diagnostics": {}, "unmatched_rows": [], "individual_rows": []}
    records = [base_report(0, "Unknown Report")]
    attach_report_fields(records, rollup_result, FIELD_EXPORT_MODE_REPORT_MAPPING, cfg)
    assert records[0]["field_extraction_status"] == STATUS_NO_FIELDS
    assert records[0]["report_fields_set"] == set()


def test_attach_catalog_only_mode(cfg):
    records = [base_report(0)]
    attach_report_fields(records, {}, FIELD_EXPORT_MODE_CATALOG_ONLY, cfg)
    assert records[0]["field_extraction_status"] == STATUS_CATALOG_ONLY
    assert records[0]["report_fields_set"] == set()


def test_attach_missing_mode(cfg):
    records = [base_report(0)]
    attach_report_fields(records, {}, FIELD_EXPORT_MODE_MISSING, cfg)
    assert records[0]["field_extraction_status"] == STATUS_NOT_PROVIDED


# ---- Duplicate detection with field extraction status ----------------------

def test_dedup_with_both_fields_confirmed(cfg):
    r0 = base_report(0, "Annual Performance")
    r1 = base_report(1, "Annual Performance Review")
    fs = {"worker|employee id||", "worker|manager||", "worker|rating||"}
    r0["report_fields_set"] = fs
    r0["field_extraction_status"] = STATUS_MATCHED
    r1["report_fields_set"] = fs
    r1["field_extraction_status"] = STATUS_MATCHED
    groups, meta_only = detect_duplicates([r0, r1], cfg)
    assert len(groups) == 1
    assert meta_only == []


def test_dedup_one_report_missing_fields_goes_to_meta_only(cfg):
    r0 = base_report(0, "Annual Performance")
    r1 = base_report(1, "Annual Performance Review")
    r0["report_fields_set"] = {"worker|employee id||", "worker|manager||"}
    r0["field_extraction_status"] = STATUS_MATCHED
    r1["report_fields_set"] = set()
    r1["field_extraction_status"] = STATUS_NO_FIELDS
    groups, meta_only = detect_duplicates([r0, r1], cfg)
    assert groups == []
    assert len(meta_only) == 1


# ---- Keeper selection when field extraction is unavailable -----------------

def test_keeper_falls_through_to_risk_when_fields_unavailable(cfg):
    """Both records lack field extraction — keeper is chosen by lower risk score."""
    rich = dict(report_uid=0, report_name="R0", report_fields_set=set(),
                field_extraction_status=STATUS_NO_FIELDS,
                total_risk_score=40, last_run_date=pd.NaT, times_run=5,
                owner="Alice", ownership_flags=[], shared="Yes",
                areas_used="a", landing_page="l", description="d",
                category="c", report_tag="t", data_source="ds", report_prompts="p")
    lean = dict(report_uid=1, report_name="R1", report_fields_set=set(),
                field_extraction_status=STATUS_NO_FIELDS,
                total_risk_score=5, last_run_date=pd.NaT, times_run=5,
                owner="Alice", ownership_flags=[], shared="Yes",
                areas_used="a", landing_page="l", description="d",
                category="c", report_tag="t", data_source="ds", report_prompts="p")
    keeper_uid, reason = choose_suggested_keeper([rich, lean])
    # Both have effective_field_count=0; tie broken by lower risk -> lean (uid=1)
    assert keeper_uid == 1
