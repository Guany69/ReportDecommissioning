"""Hard deletion rules. A match short-circuits soft scoring entirely."""
from __future__ import annotations

from dataclasses import dataclass

from .clean import is_null, is_true, text

HARD_RULE_RECOMMENDATION = "Delete - Hard Rule Match"


@dataclass
class HardHit:
    rule_id: str
    reason: str


def _dnu_in_name(r: dict):
    return "dnu" in text(r.get("report_name")).lower()


def _orphan_worklet(r: dict):
    return (
        is_true(r.get("worklet"))
        and is_null(r.get("landing_page"))
        and is_null(r.get("areas_used"))
    )


# (rule_id, predicate, reason) — evaluated in order; first match wins.
HARD_RULES = [
    ("dnu_in_name", _dnu_in_name, "Report name contains DNU."),
    ("orphan_worklet", _orphan_worklet,
     "Report is marked as a worklet but has no landing page and no area where used."),
]


def apply_hard_rules(r: dict) -> HardHit | None:
    for rule_id, pred, reason in HARD_RULES:
        if pred(r):
            return HardHit(rule_id=rule_id, reason=reason)
    return None