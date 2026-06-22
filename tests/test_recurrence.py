"""Recurrence detection from execution timestamps only."""
import pandas as pd

from report_cleanup.recurrence import (CLASS_INSUFFICIENT, CLASS_MODERATE,
                                       CLASS_NONE, CLASS_STRONG,
                                       detect_recurrence)

BASE = pd.Timestamp("2026-01-05 09:00:00")  # a Monday


def _series(day_gaps):
    """Build timestamps from cumulative day gaps starting at BASE."""
    ts, cur = [], BASE
    for g in day_gaps:
        cur = cur + pd.Timedelta(days=g)
        ts.append(cur)
    return [BASE] + ts


def test_strong_daily(cfg):
    r = detect_recurrence([BASE + pd.Timedelta(days=i) for i in range(6)], cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "daily"
    assert r.match_percentage == 1.0


def test_strong_weekly(cfg):
    r = detect_recurrence([BASE + pd.Timedelta(days=7 * i) for i in range(5)], cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "weekly"


def test_strong_biweekly(cfg):
    r = detect_recurrence([BASE + pd.Timedelta(days=14 * i) for i in range(5)], cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "biweekly"


def test_strong_monthly(cfg):
    r = detect_recurrence([BASE + pd.Timedelta(days=30 * i) for i in range(5)], cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "monthly"


def test_monthly_month_length_variation(cfg):
    # First of each month: gaps 31, 28, 31, 30 — all within the monthly range.
    months = [pd.Timestamp(f"2026-{m:02d}-01 08:00") for m in range(1, 6)]
    r = detect_recurrence(months, cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "monthly"


def test_weekend_shifted_monthly(cfg):
    # Monthly cadence nudged by a few days each time: 28, 33, 30, 34 days.
    r = detect_recurrence(_series([28, 33, 30, 34]), cfg)
    assert r.classification == CLASS_STRONG
    assert r.cadence == "monthly"


def test_moderate_three_occurrences(cfg):
    # Exactly three weekly occurrences -> two matching intervals -> Moderate.
    r = detect_recurrence([BASE, BASE + pd.Timedelta(days=7), BASE + pd.Timedelta(days=14)], cfg)
    assert r.classification == CLASS_MODERATE
    assert r.cadence == "weekly"
    assert r.occurrences == 3


def test_insufficient_history(cfg):
    r = detect_recurrence([BASE, BASE + pd.Timedelta(days=7)], cfg)
    assert r.classification == CLASS_INSUFFICIENT
    assert r.occurrences == 2


def test_irregular_no_pattern(cfg):
    # Gaps 3, 47, 40 days — no single cadence covers >=75% and not all agree.
    r = detect_recurrence(_series([3, 47, 40]), cfg)
    assert r.classification == CLASS_NONE


def test_burst_collapsing(cfg):
    # A burst of three runs within 30 minutes counts as ONE occurrence, leaving
    # three weekly occurrences -> Moderate (not inflated to Strong).
    burst = [BASE, BASE + pd.Timedelta(minutes=5), BASE + pd.Timedelta(minutes=20)]
    later = [BASE + pd.Timedelta(days=7), BASE + pd.Timedelta(days=14)]
    r = detect_recurrence(burst + later, cfg)
    assert r.occurrences == 3
    assert r.classification == CLASS_MODERATE


def test_invalid_timestamps_excluded(cfg):
    # None / NaT / unparseable values are dropped; valid daily series still detected.
    ts = [BASE, None, pd.NaT, "not-a-date", BASE + pd.Timedelta(days=1),
          BASE + pd.Timedelta(days=2), BASE + pd.Timedelta(days=3)]
    r = detect_recurrence(ts, cfg)
    assert r.occurrences == 4
    assert r.classification == CLASS_STRONG
    assert r.cadence == "daily"


def test_too_few_after_dropping_invalid(cfg):
    r = detect_recurrence([BASE, "bad", None], cfg)
    assert r.classification == CLASS_INSUFFICIENT


def test_six_month_window_does_not_force_nonrecurring(cfg):
    # Only ~6 months of weekly data still classifies as recurring (we never infer
    # 'non-recurring' from the limited window).
    r = detect_recurrence([BASE + pd.Timedelta(days=7 * i) for i in range(20)], cfg)
    assert r.classification == CLASS_STRONG
