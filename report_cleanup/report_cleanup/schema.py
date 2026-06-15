"""Canonical fields + header auto-mapping.

Source exports have messy, inconsistent headers. We map each source column to a
canonical internal key (e.g. "Number of times" -> times_run) using the alias
lists in config.yaml: exact normalized-alias match first, then containment.
"""
from __future__ import annotations

import re

# Canonical fields belonging to each table (drives validation + mapping scope).
TABLE1_FIELDS = [
    "report_name", "report_tag", "category", "landing_page", "owner", "created_by",
    "description", "created_date", "times_run", "last_updated", "last_run_date",
    "last_run_by", "data_source", "report_type", "worklet", "chart_type", "shared",
    "report_prompts", "areas_used", "owner_active",
    # NOTE: report_fields is intentionally absent — field data now comes exclusively
    # from table3 (the field export). See field_rollup.py.
]
TABLE2_FIELDS = [
    "t2_report_name", "t2_report_type", "t2_report_owner", "t2_start_date",
    "t2_requested_by", "t2_requested_id", "t2_exec_mode", "t2_exec_type",
]
TABLE3_FIELDS = [
    "t3_report_name", "t3_report_id", "t3_where_used",
    "t3_business_object", "t3_field_name", "t3_description",
    "t3_report_field_type", "t3_related_business_object",
    "t3_built_in_prompts", "t3_domain", "t3_categories", "t3_authorized_usage",
]

# Fields whose absence only degrades scoring (warn, do not abort).
SCORING_FIELDS = [
    "last_run_date", "times_run", "last_updated", "worklet", "areas_used", "shared",
    "owner", "created_by", "category", "report_tag", "data_source", "description",
    "created_date", "landing_page",
]

# How each canonical field should be cleaned (see clean.py).
FIELD_TYPES = {
    "created_date": "date", "last_updated": "date", "last_run_date": "date",
    "t2_start_date": "date",
    "times_run": "number",
    "worklet": "bool", "shared": "bool",
}


def normalize_header(h: str) -> str:
    """Lowercase, unify separators (space/_/-/slash) to single spaces, trim."""
    return re.sub(r"[\s_\-/]+", " ", str(h).strip().lower()).strip()


def auto_map(headers, aliases: dict[str, list[str]], fields: list[str]) -> dict[str, str]:
    """Return canonical_key -> source_header for the given field scope.

    A source header is claimed by at most one canonical field (first match
    wins, in `fields` order) so two keys never fight over the same column.
    """
    norm_headers = [(h, normalize_header(h)) for h in headers]
    claimed: set[str] = set()
    mapping: dict[str, str] = {}

    def claim(header: str, key: str):
        mapping[key] = header
        claimed.add(header)

    # Pass 1: exact normalized alias match.
    for key in fields:
        for alias in aliases.get(key, []):
            a = normalize_header(alias)
            hit = next((h for h, n in norm_headers if n == a and h not in claimed), None)
            if hit:
                claim(hit, key)
                break

    # Pass 2: containment for still-unmapped fields.
    for key in fields:
        if key in mapping:
            continue
        for alias in aliases.get(key, []):
            a = normalize_header(alias)
            hit = next(
                (h for h, n in norm_headers
                 if h not in claimed and (a in n or n in a)),
                None,
            )
            if hit:
                claim(hit, key)
                break

    return mapping
