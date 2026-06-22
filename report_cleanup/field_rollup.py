"""Extract report field data from the Fields export (table3) via Where_Used.

Each Fields row is ONE unique field — we never merge rows or judge whether two
rows describe "the same" field. Every source row is assigned a stable field ID
(its source-row index). A row's Where_Used cell lists the reports that use that
field, one per line; we split on line breaks, exact-normalize each report name
(clean.normalize_report_name), and add the field ID to every matching report's
field set. Two reports therefore "share a field" only when the same Fields row
listed both of them.

For backward compatibility with older field exports that carry an explicit
Report Name / Report ID column, those are honored as a fallback when no
Where_Used column is present.

Entry points used by pipeline.py:
  validate_field_table()      — decide mode before cleaning
  build_report_field_rollup() — aggregate t3 rows into per-report field sets
  attach_report_fields()      — stamp field sets onto the main report records
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field

import pandas as pd

from .clean import is_null, normalize_report_name, text

# ---- Mode constants -------------------------------------------------------
FIELD_EXPORT_MODE_MISSING = "missing"
FIELD_EXPORT_MODE_REPORT_MAPPING = "report_field_mapping"
FIELD_EXPORT_MODE_CATALOG_ONLY = "catalog_only"

# Status values stamped on each report record.
STATUS_MATCHED = "Matched"
STATUS_NO_FIELDS = "No fields found"
STATUS_CATALOG_ONLY = "Catalog only - no report mapping"
STATUS_NOT_PROVIDED = "No field export provided"


# ---- Validation -----------------------------------------------------------
@dataclass
class FieldTableValidation:
    mode: str
    warnings: list[str] = dc_field(default_factory=list)
    errors: list[str] = dc_field(default_factory=list)

    def has_fatal(self) -> bool:
        return bool(self.errors)


def validate_field_table(t3_raw, t3_map: dict) -> FieldTableValidation:
    """Determine the field export mode and surface any warnings."""
    if t3_raw is None or len(t3_raw) == 0:
        return FieldTableValidation(
            mode=FIELD_EXPORT_MODE_MISSING,
            warnings=["No field export provided — field-based duplicate detection disabled."],
        )

    vr = FieldTableValidation(mode=FIELD_EXPORT_MODE_MISSING)

    if "t3_field_name" not in t3_map:
        vr.warnings.append(
            "Field export has no mappable 'Field Name' column — "
            "field analysis unavailable."
        )

    has_report_key = any(k in t3_map for k in ("t3_where_used", "t3_report_name", "t3_report_id"))

    if not has_report_key:
        vr.mode = FIELD_EXPORT_MODE_CATALOG_ONLY
        vr.warnings.append(
            "Field export has no Where_Used, Report Name, or Report ID column — "
            "treating as a field catalog. Fields cannot be mapped to individual reports; "
            "field-based duplicate detection is disabled."
        )
    else:
        vr.mode = FIELD_EXPORT_MODE_REPORT_MAPPING

    return vr


# ---- Key normalization ----------------------------------------------------
def _norm_report_key(v) -> str:
    """EXACT report-name key (case-insensitive, whitespace-collapsed)."""
    return normalize_report_name(v)


def _norm_report_id(v) -> str:
    return text(v).lower().strip()


# ---- Where_Used parser ----------------------------------------------------
# Report names inside Where_Used are separated by line breaks only. Support
# \n, \r\n, and \r — nothing else (commas/semicolons can appear inside names).
_WHERE_USED_SPLIT = re.compile(r"\r\n|\r|\n")


def parse_where_used(where_used_text: str) -> list[str]:
    """Split a Where_Used cell into a list of EXACT-normalized report name keys.

    Blank lines are dropped; duplicates within the cell are collapsed.
    """
    if is_null(where_used_text):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in _WHERE_USED_SPLIT.split(text(where_used_text)):
        key = _norm_report_key(part)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---- Field rollup ---------------------------------------------------------
def build_report_field_rollup(t3: pd.DataFrame, cfg) -> dict:
    """Aggregate the canonical field-export DataFrame into one entry per report key.

    Returns a dict with:
      rollup          — {report_key: {field_set, field_count, field_list_text,
                                       field_signature, business_objects/built_in_prompts/
                                       related_bos/authorized_usage/domains/categories sets,
                                       match_type}}
      diagnostics     — counters for the diagnostics tab
      unmatched_rows  — Fields rows that produced no report key at all
      individual_rows — one dict per (report_key, field_id) link for DB storage
    """
    fields_cfg = cfg.get("fields") or {}
    prefer_id = fields_cfg.get("prefer_report_id", True)
    allow_where_used = fields_cfg.get("allow_where_used_parsing", True)

    agg: dict[str, dict] = {}
    unmatched_rows: list[dict] = []
    individual_rows: list[dict] = []

    diag: dict[str, int] = {
        "total_field_rows": len(t3),
        "total_report_field_links": 0,
        "total_unmatched_rows": 0,
        "total_where_used_rows_parsed": 0,
        "total_where_used_parse_failures": 0,
        "blank_where_used_rows": 0,
        "multi_report_field_rows": 0,
        "reports_with_duplicate_field_keys": 0,
    }

    # to_dict("records") is far faster than iterrows() (no per-row Series) — matters
    # when the Fields export is large.
    for idx, row_d in enumerate(t3.to_dict("records")):
        # Stable per-row field ID — never merged with any other row.
        field_id = f"f{idx}"
        field_display = text(row_d.get("t3_field_name", "")) or field_id
        business_object = text(row_d.get("t3_business_object", ""))
        related_bo = text(row_d.get("t3_related_business_object", ""))
        report_field_type = text(row_d.get("t3_report_field_type", ""))
        built_in_prompts = text(row_d.get("t3_built_in_prompts", ""))
        domain = text(row_d.get("t3_domain", ""))
        categories = text(row_d.get("t3_categories", ""))
        authorized_usage = text(row_d.get("t3_authorized_usage", ""))

        # Determine report key(s): Where_Used first (the real linkage), then the
        # legacy Report ID / Report Name fallbacks for older exports.
        report_keys: list[str] = []
        match_type = ""

        where_used_val = text(row_d.get("t3_where_used", ""))
        report_id_val = text(row_d.get("t3_report_id", ""))
        report_name_val = text(row_d.get("t3_report_name", ""))

        if allow_where_used and "t3_where_used" in row_d and where_used_val:
            diag["total_where_used_rows_parsed"] += 1
            keys = parse_where_used(where_used_val)
            if keys:
                report_keys = keys
                match_type = "where_used"
                if len(keys) > 1:
                    diag["multi_report_field_rows"] += 1
            else:
                diag["total_where_used_parse_failures"] += 1
        elif allow_where_used and "t3_where_used" in row_d and not where_used_val:
            diag["blank_where_used_rows"] += 1

        if not report_keys and prefer_id and report_id_val:
            rk = _norm_report_id(report_id_val)
            if rk:
                report_keys, match_type = [rk], "report_id"
        if not report_keys and report_name_val:
            rk = _norm_report_key(report_name_val)
            if rk:
                report_keys, match_type = [rk], "report_name"

        if not report_keys:
            unmatched_rows.append({**row_d, "_reason": "no_report_key", "_field_id": field_id})
            diag["total_unmatched_rows"] += 1
            continue

        for rk in report_keys:
            acc = agg.setdefault(rk, {
                "field_set": set(), "field_display_names": set(),
                "business_objects": set(), "built_in_prompts": set(),
                "related_bos": set(), "authorized_usage": set(),
                "domains": set(), "categories": set(), "match_type": match_type,
            })
            if field_id in acc["field_set"]:
                diag["reports_with_duplicate_field_keys"] += 1
            acc["field_set"].add(field_id)
            acc["field_display_names"].add(field_display)
            if business_object:
                acc["business_objects"].add(business_object.casefold())
            if built_in_prompts:
                acc["built_in_prompts"].add(built_in_prompts.casefold())
            if related_bo:
                acc["related_bos"].add(related_bo.casefold())
            if authorized_usage:
                acc["authorized_usage"].add(authorized_usage.casefold())
            if domain:
                acc["domains"].add(domain)
            if categories:
                acc["categories"].add(categories)
            diag["total_report_field_links"] += 1

            individual_rows.append({
                "report_key": rk, "field_key": field_id, "field_name": field_display,
                "business_object": business_object, "related_business_object": related_bo,
                "report_field_type": report_field_type, "domain": domain,
                "categories": categories, "authorized_usage": authorized_usage,
                "source_match_type": match_type,
            })

    rollup: dict[str, dict] = {}
    for rk, acc in agg.items():
        fs = acc["field_set"]
        rollup[rk] = {
            "field_set": fs,
            "field_count": len(fs),
            "field_list_text": "; ".join(sorted(acc["field_display_names"])),
            "field_signature": "|".join(sorted(fs)),
            "business_objects_set": acc["business_objects"],
            "built_in_prompts_set": acc["built_in_prompts"],
            "related_bos_set": acc["related_bos"],
            "authorized_usage_set": acc["authorized_usage"],
            "business_objects_used": sorted(acc["business_objects"]),
            "domains_used": sorted(acc["domains"]),
            "categories_used": sorted(acc["categories"]),
            "match_type": acc["match_type"],
        }

    return {
        "rollup": rollup,
        "diagnostics": diag,
        "unmatched_rows": unmatched_rows,
        "individual_rows": individual_rows,
    }


# ---- Attach to report records --------------------------------------------
def attach_report_fields(
    records: list[dict],
    field_rollup_result: dict,
    field_mode: str,
    cfg,
) -> list[dict]:
    """Stamp field-set data from the rollup onto each report record (matched by
    EXACT normalized report name, with a report_id fallback). Also records, in the
    rollup diagnostics, which Where_Used report keys never matched a Comprehensive
    report (unmatched_where_used_names) and per-report fields_mapped counts."""
    fields_cfg = cfg.get("fields") or {}
    prefer_id = fields_cfg.get("prefer_report_id", True)

    def _blank(r: dict, status: str) -> None:
        r["report_fields_set"] = set()
        r["field_count"] = 0
        r["report_fields"] = None
        r["field_signature"] = None
        r["business_objects_used"] = []
        r["domains_used"] = []
        r["categories_used"] = []
        r["business_objects_set"] = set()
        r["built_in_prompts_set"] = set()
        r["related_bos_set"] = set()
        r["authorized_usage_set"] = set()
        r["field_extraction_status"] = status

    if field_mode == FIELD_EXPORT_MODE_MISSING:
        for r in records:
            _blank(r, STATUS_NOT_PROVIDED)
        return records

    if field_mode == FIELD_EXPORT_MODE_CATALOG_ONLY:
        for r in records:
            _blank(r, STATUS_CATALOG_ONLY)
        return records

    rollup = field_rollup_result.get("rollup", {})
    matched_keys: set[str] = set()

    for r in records:
        report_id_val = text(r.get("report_id", ""))
        report_name_val = text(r.get("report_name", ""))

        rk: str | None = None
        if prefer_id and report_id_val:
            candidate = _norm_report_id(report_id_val)
            if candidate in rollup:
                rk = candidate
        if rk is None:
            candidate = _norm_report_key(report_name_val)
            if candidate in rollup:
                rk = candidate

        entry = rollup.get(rk) if rk else None
        if entry:
            matched_keys.add(rk)
            r["report_fields_set"] = entry.get("field_set", set())
            r["field_count"] = entry.get("field_count", len(entry.get("field_set", set())))
            r["report_fields"] = entry.get("field_list_text")
            r["field_signature"] = entry.get("field_signature")
            r["business_objects_used"] = entry.get("business_objects_used", [])
            r["domains_used"] = entry.get("domains_used", [])
            r["categories_used"] = entry.get("categories_used", [])
            r["business_objects_set"] = entry.get("business_objects_set", set())
            r["built_in_prompts_set"] = entry.get("built_in_prompts_set", set())
            r["related_bos_set"] = entry.get("related_bos_set", set())
            r["authorized_usage_set"] = entry.get("authorized_usage_set", set())
            r["field_extraction_status"] = STATUS_MATCHED
        else:
            _blank(r, STATUS_NO_FIELDS)

    # Where_Used report keys that no Comprehensive report claimed.
    unmatched = sorted(set(rollup) - matched_keys)
    diag = field_rollup_result.setdefault("diagnostics", {})
    diag["unmatched_where_used_reports"] = len(unmatched)
    field_rollup_result["unmatched_where_used_names"] = unmatched

    return records
