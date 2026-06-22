"""Assemble the single Overall Decommissioning Score (0..100).

    cleanup_percentage = cleanup_risk_points / cleanup_risk_max * 100
    overall_score      = round(cleanup_percentage - business_protection_credit)
    overall_score      = clamp(overall_score, 0, soft_score_cap)   # 99 by default
    if hard_rule_triggered: overall_score = hard_rule_score          # 100

Duplicate evidence is never an input here. The reason trail records every term so
the score is fully reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .soft_scoring import Reason


@dataclass
class OverallResult:
    overall_score: int
    cleanup_risk_points: int
    cleanup_risk_max: int
    cleanup_percentage: int
    business_protection_credit: int
    hard_rule_triggered: bool
    reasons: list[Reason] = field(default_factory=list)


def calculate_overall_score(
    cleanup_points: int,
    protection_credit: int,
    hard_rule_triggered: bool,
    cfg,
) -> OverallResult:
    o = cfg.overall
    cmax = o["cleanup_risk_max"]
    cap = o["soft_score_cap"]
    hard_score = o["hard_rule_score"]

    cleanup_pct_f = (cleanup_points / cmax * 100) if cmax else 0.0
    cleanup_pct = round(cleanup_pct_f)

    reasons: list[Reason] = [
        Reason("overall", f"Cleanup risk earned {cleanup_points} of {cmax} points", cleanup_points),
        Reason("overall", f"Cleanup percentage {cleanup_pct}%", None),
    ]

    if hard_rule_triggered:
        score = hard_score
        credit_applied = 0
        reasons.append(Reason("overall", f"Hard-rule override — Overall Score set to {hard_score}", None))
    else:
        if protection_credit:
            reasons.append(Reason("overall", f"Total business-protection credit {protection_credit}", -protection_credit))
        score = max(0, min(cap, round(cleanup_pct_f - protection_credit)))
        credit_applied = protection_credit
        reasons.append(Reason("overall", f"Final Overall Score {score}", None))

    return OverallResult(
        overall_score=score,
        cleanup_risk_points=cleanup_points,
        cleanup_risk_max=cmax,
        cleanup_percentage=cleanup_pct,
        business_protection_credit=credit_applied,
        hard_rule_triggered=hard_rule_triggered,
        reasons=reasons,
    )
