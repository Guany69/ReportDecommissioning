"""Report cleanup *risk* scoring (0..60) with a full reason trail.

Risk answers a single question: "how risky / cleanup-worthy is this report?"
It is derived ONLY from direct report fields (last run, times run, last updated,
areas used, landing page, shared). Execution-table presence, ownership and data
quality no longer affect the score — ownership/quality are surfaced as flags
(see flags.py), not points.

Rubric is encoded in config.yaml. Two correctness invariants:
  * Usage Risk is summed then capped at +45.
  * Last-Run and Last-Updated are TIERED — only the single highest tier applies.

`today` is injectable so date-based tests are stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .clean import is_false, is_null


@dataclass
class Reason:
    category: str
    label: str
    points: int | None


@dataclass
class ScoreResult:
    total_risk_score: int
    usage_risk: int
    age_risk: int
    usage_context_risk: int
    recommendation: str
    reasons: list[Reason] = field(default_factory=list)

    # Backward-compatible aliases.
    @property
    def score(self) -> int:
        return self.total_risk_score

    @property
    def band(self) -> str:
        return self.recommendation


def _today(today=None) -> pd.Timestamp:
    return pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()


def _days_since(ts, today=None) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    return (_today(today) - pd.Timestamp(ts).normalize()).days


def _band_for(score: int, bands) -> str:
    for b in bands:
        if score <= b["max"]:
            return b["label"]
    return bands[-1]["label"]


# ---------------------------------------------------------------------------
# Component helpers (each returns (points, reasons)).
# ---------------------------------------------------------------------------
def calculate_usage_risk(report: dict, today, cfg) -> tuple[int, list[Reason]]:
    """Last-run recency + run-count, summed then capped at usage_cap (45)."""
    s = cfg.scoring
    u = s["usage"]
    reasons: list[Reason] = []
    usage = 0

    # Recency uses the EFFECTIVE last-run date (max of Comprehensive's date and the
    # latest execution seen in the six-month Runs export). Unit tests that pass only
    # last_run_date keep working via the fallback.
    if "effective_last_run_date" in report:
        last_run = report.get("effective_last_run_date")
    else:
        last_run = report.get("last_run_date")
    run_days = _days_since(last_run, today)
    if is_null(last_run):
        usage += u["last_run_null"]
        reasons.append(Reason("usage", "Effective Last Run Date is blank", u["last_run_null"]))
    elif run_days is not None and run_days > s["days"]["year3"]:
        usage += u["last_run_over_3y"]
        reasons.append(Reason("usage", "Effective Last Run Date older than 3 years", u["last_run_over_3y"]))
    elif run_days is not None and run_days > s["days"]["year2"]:
        usage += u["last_run_over_2y"]
        reasons.append(Reason("usage", "Effective Last Run Date older than 2 years", u["last_run_over_2y"]))
    elif run_days is not None and run_days > s["days"]["year1"]:
        usage += u["last_run_over_1y"]
        reasons.append(Reason("usage", "Effective Last Run Date older than 1 year", u["last_run_over_1y"]))

    # Missing/blank run count is treated as 0 (no safer numeric convention exists).
    tr = report.get("times_run")
    tr = 0.0 if (tr is None or pd.isna(tr)) else float(tr)
    if tr == 0:
        usage += u["times_run_zero"]
        reasons.append(Reason("usage", "Number of Times Run = 0", u["times_run_zero"]))
    elif 1 <= tr <= 3:
        usage += u["times_run_1_3"]
        reasons.append(Reason("usage", "Number of Times Run between 1 and 3", u["times_run_1_3"]))
    elif 4 <= tr <= 10:
        usage += u["times_run_4_10"]
        reasons.append(Reason("usage", "Number of Times Run between 4 and 10", u["times_run_4_10"]))

    cap = s["usage_cap"]
    if usage > cap:
        reasons.append(Reason("usage", f"Usage risk capped at {cap}", -(usage - cap)))
        usage = cap
    return usage, reasons


def calculate_age_risk(report: dict, today, cfg) -> tuple[int, list[Reason]]:
    """Single highest tier on Last Updated Date."""
    a = cfg.scoring["age"]
    days = cfg.scoring["days"]
    reasons: list[Reason] = []
    age = 0

    lu = report.get("last_updated")
    lu_days = _days_since(lu, today)
    if is_null(lu):
        age += a["last_updated_null"]
        reasons.append(Reason("age", "Last Updated Date is null", a["last_updated_null"]))
    elif lu_days is not None and lu_days > days["year3"]:
        age += a["last_updated_over_3y"]
        reasons.append(Reason("age", "Last Updated Date older than 3 years", a["last_updated_over_3y"]))
    elif lu_days is not None and lu_days > days["year2"]:
        age += a["last_updated_over_2y"]
        reasons.append(Reason("age", "Last Updated Date older than 2 years", a["last_updated_over_2y"]))
    return age, reasons


def calculate_usage_context_risk(report: dict, cfg) -> tuple[int, list[Reason]]:
    """Areas used / landing page / shared — these stack."""
    c = cfg.scoring["context"]
    reasons: list[Reason] = []
    ctx = 0
    if is_null(report.get("areas_used")):
        ctx += c["areas_used_null"]
        reasons.append(Reason("context", "Areas Where Used is blank", c["areas_used_null"]))
    if is_null(report.get("landing_page")):
        ctx += c["landing_page_null"]
        reasons.append(Reason("context", "Worklet Landing Page is blank", c["landing_page_null"]))
    if is_false(report.get("shared")):
        ctx += c["not_shared"]
        reasons.append(Reason("context", "Shared = No", c["not_shared"]))
    return ctx, reasons


def calculate_total_risk_score(report: dict, today=None, cfg=None) -> ScoreResult:
    usage, r1 = calculate_usage_risk(report, today, cfg)
    age, r2 = calculate_age_risk(report, today, cfg)
    ctx, r3 = calculate_usage_context_risk(report, cfg)
    total = max(0, min(60, usage + age + ctx))
    return ScoreResult(
        total_risk_score=total,
        usage_risk=usage,
        age_risk=age,
        usage_context_risk=ctx,
        recommendation=_band_for(total, cfg.bands),
        reasons=r1 + r2 + r3,
    )


def score_report(r: dict, cfg, today=None) -> ScoreResult:
    """Public entry point (kept for pipeline/back-compat)."""
    return calculate_total_risk_score(r, today, cfg)
