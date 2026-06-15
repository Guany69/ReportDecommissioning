"""Extract report field data from a third-party field / data-dictionary export (table3).

The third export may contain either:
  (a) A report-to-field mapping — each row links a field to a specific report via
      Report Name, Report ID, or a Where_Used column.  We roll fields up per report
      and attach the field sets to the main report records so duplicate detection
      can use real field-level evidence.
  (b) A field catalog — fields are described but not linked to individual reports.
      We emit a diagnostic and skip field-based duplicate detection entirely.

Entry points used by pipeline.py:
  validate_field_table()      — decide mode before cleaning
  build_report_field_rollup() — aggregate t3 rows into per-report field sets
  attach_report_fields()      — stamp field sets onto the main report records
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field

import pandas as pd

from .clean import is_null, normalize_name, text

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

    has_report_key = any(k in t3_map for k in ("t3_report_name", "t3_report_id", "t3_where_used"))

    if not has_report_key:
        vr.mode = FIELD_EXPORT_MODE_CATALOG_ONLY
        vr.warnings.append(
            "Field export has no Report Name, Report ID, or Where_Used column — "
            "treating as a field catalog. Fields cannot be mapped to individual reports; "
            "field-based duplicate detection is disabled."
        )
    else:
        vr.mode = FIELD_EXPORT_MODE_REPORT_MAPPING

    return vr


# ---- Key normalization ----------------------------------------------------
def _norm_report_key(v, name_noise) -> str:
    """Normalized report name key — identical convention to join.py."""
    return normalize_name(v, name_noise)


def _norm_report_id(v) -> str:
    return text(v).lower().strip()


def _norm_field_name(v) -> str:
    s = text(v).lower()
    s = re.sub(r"[\s_]+", " ", s).strip()
    return s


def build_field_key(row: dict, field_key_parts: list[str]) -> str:
    """Stable field identifier: business_object|field_name|related_bo|field_type.

    Uses the configured field_key_parts list so operators can tune which
    columns make a field unique without changing Python code.
    """
    parts = [_norm_field_name(text(row.get(f"t3_{p}", ""))) for p in field_key_parts]
    return "|".join(parts)


# ---- Where_Used parser ----------------------------------------------------
_WHERE_USED_SPLIT = re.compile(r"[\n\r;|•·–—]+")


def parse_where_used(where_used_text: str, cfg) -> list[str]:
    """Split a Where_Used cell value into a list of normalized report name keys.

    Comma splitting is disabled by default (commas appear inside report names).
    """
    if is_null(where_used_text):
        return []

    fields_cfg = cfg.get("fields") or {}
    allow_comma = fields_cfg.get("allow_comma_split_in_where_used", False)
    name_noise = (cfg.clean or {}).get("name_noise", [])

    raw = text(where_used_text)
    parts = _WHERE_USED_SPLIT.split(raw)

    if allow_comma:
        expanded: list[str] = []
        for p in parts:
            expanded.extend(p.split(","))
        parts = expanded

    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        cleaned = p.strip().strip('"').strip("'").strip()
        if len(cleaned) < 2:
            continue
        key = _norm_report_key(cleaned, name_noise)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---- Field rollup ---------------------------------------------------------
def build_report_field_rollup(t3: pd.DataFrame, cfg) -> dict:
    """Aggregate the canonical field-export DataFrame into one entry per report key.

    Returns a dict with three keys:
      rollup        — {report_key: {field_set, field_count, field_list_text,
                                    field_signature, business_objects_used,
                                    domains_used, categories_used, match_type}}
      diagnostics   — counters for the diagnostics tab
      unmatched_rows — rows that could not be attributed to any report
      individual_rows — one dict per (report_key, field_key) pair for DB storage
    """
    fields_cfg = cfg.get("fields") or {}
    name_noise = (cfg.clean or {}).get("name_noise", [])
    prefer_id = fields_cfg.get("prefer_report_id", True)
    allow_where_used = fields_cfg.get("allow_where_used_parsing", True)
    field_key_parts = fields_cfg.get(
        "field_key_parts",
        ["business_object", "field_name", "related_business_object", "report_field_type"],
    )

    agg: dict[str, dict] = {}
    unmatched_rows: list[dict] = []
    individual_rows: list[dict] = []

    diag: dict[str, int] = {
        "total_field_rows": len(t3),
        "total_report_field_links": 0,
        "total_unmatched_rows": 0,
        "total_where_used_rows_parsed": 0,
        "total_where_used_parse_failures": 0,
        "ambiguous_report_matches": 0,
        "reports_with_duplicate_field_keys": 0,
    }

    for _, row in t3.iterrows():
        row_d = row.to_dict()
        field_key = build_field_key(row_d, field_key_parts)
        field_display = text(row_d.get("t3_field_name", ""))
        business_object = text(row_d.get("t3_business_object", ""))
        related_bo = text(row_d.get("t3_related_business_object", ""))
        report_field_type = text(row_d.get("t3_report_field_type", ""))
        domain = text(row_d.get("t3_domain", ""))
        categories = text(row_d.get("t3_categories", ""))
        authorized_usage = text(row_d.get("t3_authorized_usage", ""))

        # Determine report key(s) — prefer report_id, then report_name, then where_used.
        report_keys: list[str] = []
        match_type = ""

        report_id_val = text(row_d.get("t3_report_id", ""))
        report_name_val = text(row_d.get("t3_report_name", ""))
        where_used_val = text(row_d.get("t3_where_used", ""))

        if prefer_id and report_id_val:
            rk = _norm_report_id(report_id_val)
            if rk:
                report_keys = [rk]
                match_type = "report_id"
        if not report_keys and report_name_val:
            rk = _norm_report_key(report_name_val, name_noise)
            if rk:
                report_keys = [rk]
                match_type = "report_name"
        if not report_keys and allow_where_used and where_used_val:
            diag["total_where_used_rows_parsed"] += 1
            keys = parse_where_used(where_used_val, cfg)
            if keys:
                report_keys = keys
                match_type = "where_used"
                if len(keys) > 1:
                    diag["ambiguous_report_matches"] += 1
            else:
                diag["total_where_used_parse_failures"] += 1

        if not report_keys or not field_key:
            reason = "no_report_key" if not report_keys else "no_field_key"
            unmatched_rows.append({**row_d, "_reason": reason})
            diag["total_unmatched_rows"] += 1
            continue

        for rk in report_keys:
            if not rk:
                continue
            acc = agg.setdefault(rk, {
                "field_set": set(), "field_display_names": set(),
                "business_objects": set(), "domains": set(), "categories": set(),
                "match_type": match_type,
            })
            # Track duplicate field keys per report (diagnostic only).
            if field_key in acc["field_set"]:
                diag["reports_with_duplicate_field_keys"] += 1
            acc["field_set"].add(field_key)
            if field_display:
                acc["field_display_names"].add(field_display)
            if business_object:
                acc["business_objects"].add(business_object)
            if domain:
                acc["domains"].add(domain)
            if categories:
                acc["categories"].add(categories)
            diag["total_report_field_links"] += 1

            individual_rows.append({
                "report_key": rk,
                "field_key": field_key,
                "field_name": field_display,
                "business_object": business_object,
                "related_business_object": related_bo,
                "report_field_type": report_field_type,
                "domain": domain,
                "categories": categories,
                "authorized_usage": authorized_usage,
                "source_match_type": match_type,
            })

    # Finalize rollup entries.
    rollup: dict[str, dict] = {}
    for rk, acc in agg.items():
        fs = acc["field_set"]
        rollup[rk] = {
            "field_set": fs,
            "field_count": len(fs),
            "field_list_text": "; ".join(sorted(acc["field_display_names"])),
            "field_signature": "|".join(sorted(fs)),
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
    """Stamp field-set data from the rollup onto each report record.

    Sets on each record (always, regardless of mode):
      report_fields_set       set[str]   — used by dedup.detect_duplicates
      report_fields           str | None — semicolon-separated display names for Excel/DB
      field_count             int
      field_signature         str | None — pipe-separated sorted field keys
      business_objects_used   list[str]
      domains_used            list[str]
      categories_used         list[str]
      field_extraction_status str        — STATUS_* constant
    """
    name_noise = (cfg.clean or {}).get("name_noise", [])
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

    for r in records:
        report_id_val = text(r.get("report_id", ""))
        report_name_val = text(r.get("report_name", ""))

        rk: str | None = None
        if prefer_id and report_id_val:
            candidate = _norm_report_id(report_id_val)
            if candidate in rollup:
                rk = candidate
        if rk is None:
            candidate = _norm_report_key(report_name_val, name_noise)
            if candidate in rollup:
                rk = candidate

        entry = rollup.get(rk) if rk else None
        if entry:
            r["report_fields_set"] = entry["field_set"]
            r["field_count"] = entry["field_count"]
            r["report_fields"] = entry["field_list_text"]
            r["field_signature"] = entry["field_signature"]
            r["business_objects_used"] = entry["business_objects_used"]
            r["domains_used"] = entry["domains_used"]
            r["categories_used"] = entry["categories_used"]
            r["field_extraction_status"] = STATUS_MATCHED
        else:
            _blank(r, STATUS_NO_FIELDS)

    return records
