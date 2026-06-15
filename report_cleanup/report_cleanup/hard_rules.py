"""Hard deletion rules. A match overrides all soft scoring and pins the Overall
Decommissioning Score to 100 (reserved exclusively for hard-rule matches)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clean import is_null, is_true, text

HARD_RULE_RECOMMENDATION = "Automatic Decommissioning Recommendation"
HARD_RULE_SCORE = 100

_DNU_WORD = re.compile(r"\bdnu\b", re.IGNORECASE)


@dataclass
class HardHit:
    rule_id: str
    name: str
    reason: str


def _hr_cfg(cfg) -> dict:
    """Hard-rule settings with safe defaults when config is absent."""
    if cfg is None:
        return {}
    try:
        return cfg.get("hard_rules") or {}
    except AttributeError:
        return cfg.get("hard_rules", {}) if isinstance(cfg, dict) else {}


def _dnu_in_name(r: dict, hr: dict) -> bool:
    name = text(r.get("report_name"))
    if not name:
        return False
    # Word-boundary match by default so "DNU" doesn't fire inside unrelated words
    # (e.g. "abundance"). Operators can opt into raw substring via config.
    if hr.get("dnu", {}).get("word_boundary", True):
        return bool(_DNU_WORD.search(name))
    return "dnu" in name.lower()


def _orphan_worklet(r: dict, hr: dict) -> bool:
    # Final documented rule: a worklet is orphaned when it is a worklet AND has
    # no Worklet Landing Pages AND no standalone Landing Page AND no Area Where
    # Used — i.e. it is placed nowhere a user could reach it.
    return (
        is_true(r.get("worklet"))
        and is_null(r.get("landing_page"))
        and is_null(r.get("report_landing_page"))
        and is_null(r.get("areas_used"))
    )


# (rule_id, display name, predicate, reason, config_key) — first match wins.
HARD_RULES = [
    ("dnu_in_name", "DNU in Name", _dnu_in_name,
     "Report name contains DNU.", "dnu"),
    ("orphan_worklet", "Orphaned Worklet", _orphan_worklet,
     "Report is marked as a worklet but has no landing page and no area where used.",
     "orphan_worklet"),
]


def apply_hard_rules(r: dict, cfg=None) -> HardHit | None:
    hr = _hr_cfg(cfg)
    for rule_id, name, pred, reason, key in HARD_RULES:
        if hr.get(key, {}).get("enabled", True) is False:
            continue
        if pred(r, hr):
            return HardHit(rule_id=rule_id, name=name, reason=reason)
    return None


def has_recent_or_recurring_usage(r: dict, cfg=None) -> bool:
    """True when a (hard-ruled) report still shows live usage — used to attach a
    conflict warning. Recent = effective last run within the configured window;
    recurring = a Strong/Moderate recurrence classification."""
    recurrence = (r.get("recurrence") or {})
    if str(recurrence.get("classification", "")).lower() in {"strong", "moderate"}:
        return True
    runs = r.get("runs_exec_count") or 0
    return runs > 0
