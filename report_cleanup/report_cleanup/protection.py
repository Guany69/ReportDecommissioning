"""Business-protection credits — points SUBTRACTED from the cleanup percentage so
heavily-used, broadly-requested, or well-placed reports are not over-flagged.

Credits are explainable: every tier that fires emits a Reason carrying a negative
point value. Recurrence and requester breadth each contribute at most one tier;
business-context credits stack. The total is capped at business_protection.max_credit.

Recurrence classification is supplied by recurrence.py (Phase 2); when it is absent
the recurrence tier simply contributes nothing.
"""
from __future__ import annotations

import re

from .clean import is_true, text
from .soft_scoring import Reason

_AREA_SPLIT = re.compile(r"[,;|\n\r]+")


def _distinct_area_count(areas_used) -> int:
    raw = text(areas_used)
    if not raw:
        return 0
    parts = {p.strip().casefold() for p in _AREA_SPLIT.split(raw) if p.strip()}
    return len(parts)


def _has(report: dict, key: str) -> bool:
    return bool(text(report.get(key)))


def calculate_protection_credit(report: dict, recurrence: dict | None, cfg) -> tuple[int, list[Reason]]:
    """Return (total_credit, reasons). total_credit is a non-negative int capped
    at business_protection.max_credit; reasons carry negative points."""
    bp = cfg.business_protection
    reasons: list[Reason] = []
    credit = 0

    # -- recurrence (one tier) ----------------------------------------------
    cls = str((recurrence or {}).get("classification", "")).strip().lower()
    rec_w = bp.get("recurrence", {})
    if cls == "strong":
        credit += rec_w.get("strong", 0)
        reasons.append(Reason("protection", "Strong recurring execution pattern", -rec_w.get("strong", 0)))
    elif cls == "moderate":
        credit += rec_w.get("moderate", 0)
        reasons.append(Reason("protection", "Moderate recurring execution pattern", -rec_w.get("moderate", 0)))

    # -- requester breadth (one tier) ---------------------------------------
    dr = report.get("distinct_requesters") or 0
    req_w = bp.get("requesters", {})
    if dr >= 5:
        credit += req_w.get("five_or_more", 0)
        reasons.append(Reason("protection", f"Five or more distinct requesters ({dr})", -req_w.get("five_or_more", 0)))
    elif dr >= 2:
        credit += req_w.get("two_to_four", 0)
        reasons.append(Reason("protection", f"Two to four distinct requesters ({dr})", -req_w.get("two_to_four", 0)))

    # -- business context (stack) -------------------------------------------
    ctx_w = bp.get("context", {})
    if _distinct_area_count(report.get("areas_used")) > 1:
        credit += ctx_w.get("multiple_areas", 0)
        reasons.append(Reason("protection", "Used in multiple distinct Areas Where Used", -ctx_w.get("multiple_areas", 0)))
    active_placement = (
        is_true(report.get("worklet"))
        or _has(report, "landing_page")
        or _has(report, "report_landing_page")
    )
    if active_placement:
        credit += ctx_w.get("active_placement", 0)
        reasons.append(Reason("protection", "Active worklet or landing-page placement", -ctx_w.get("active_placement", 0)))
    if is_true(report.get("shared")):
        credit += ctx_w.get("shared_yes", 0)
        reasons.append(Reason("protection", "Shared = Yes", -ctx_w.get("shared_yes", 0)))

    # -- cap -----------------------------------------------------------------
    cap = bp.get("max_credit", credit)
    if credit > cap:
        reasons.append(Reason("protection", f"Business-protection credit capped at {cap}", (credit - cap)))
        credit = cap

    return credit, reasons
