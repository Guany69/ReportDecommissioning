"""Ownership & data-quality flags — surfaced to reviewers, NO score weight.

These used to add points to the risk score. They no longer do: cleanup risk and
data hygiene are separate concerns. Each generator returns a list of short flag
labels; the engine joins them with "; " for CSV/Excel output.
"""
from __future__ import annotations

from .clean import is_null, text


def _owner_inactive(report: dict, cfg) -> bool:
    """True only when we have real evidence the owner is inactive.

    Preference order:
      1. An explicit owner-status column (owner_active / owner_status), if present.
      2. An optional keyword heuristic on the owner name (off by default).
    With no status column and the heuristic disabled, this returns False so the
    flag stays empty — it "safely does nothing".
    """
    f = cfg.flags if cfg is not None else {}
    status = report.get("owner_active")
    if status is None:
        status = report.get("owner_status")
    if not is_null(status):
        inactive_vals = {v.lower() for v in f.get("inactive_status_values", [])}
        return text(status).lower() in inactive_vals
    if f.get("use_owner_inactive_heuristic"):
        val = text(report.get("owner")).lower()
        return any(k in val for k in f.get("inactive_keywords", []))
    return False


def generate_ownership_flags(report: dict, cfg=None) -> list[str]:
    flags: list[str] = []
    owner = report.get("owner")
    if is_null(owner):
        flags.append("Missing Owner")
    t2_owner = report.get("t2_owner")
    if not is_null(owner) and not is_null(t2_owner) and text(owner).lower() != text(t2_owner).lower():
        flags.append("Owner Mismatch")
    if is_null(report.get("created_by")):
        flags.append("Missing Creator")
    if _owner_inactive(report, cfg):
        flags.append("Inactive Owner")
    return flags


def generate_data_quality_flags(report: dict, cfg=None) -> list[str]:
    flags: list[str] = []
    if is_null(report.get("description")):
        flags.append("Missing Description")
    if is_null(report.get("category")):
        flags.append("Missing Category")
    if is_null(report.get("report_tag")):
        flags.append("Missing Report Tag")
    if is_null(report.get("data_source")):
        flags.append("Missing Data Source")
    return flags


def build_all_flags(report: dict, cfg=None) -> dict[str, list[str]]:
    """Attach ownership_flags / data_quality_flags / all_flags as lists."""
    own = generate_ownership_flags(report, cfg)
    dq = generate_data_quality_flags(report, cfg)
    return {
        "ownership_flags": own,
        "data_quality_flags": dq,
        "all_flags": own + dq,
    }
