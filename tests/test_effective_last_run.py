"""effective_last_run_date = max(Comprehensive last run, latest Runs execution)."""
import pandas as pd

from report_cleanup.exec_rollup import build_exec_rollup
from report_cleanup.join import derive_effective_last_run, join_reports


def _record(comp_last, runs_starts=(), runs_last=()):
    t1 = pd.DataFrame({"report_name": ["R"], "last_run_date": [comp_last], "times_run": [1]})
    rows = []
    for s in runs_starts:
        rows.append({"t2_report_name": "R", "t2_start_date": s, "t2_last_run_date": pd.NaT})
    for l in runs_last:
        rows.append({"t2_report_name": "R", "t2_start_date": pd.NaT, "t2_last_run_date": l})
    t2 = pd.DataFrame(rows) if rows else pd.DataFrame()
    rollup = build_exec_rollup(t2)
    records, _ = join_reports(t1, rollup)
    derive_effective_last_run(records)
    return records[0]


def test_comprehensive_newer_wins():
    r = _record(pd.Timestamp("2025-06-01"), runs_starts=[pd.Timestamp("2025-01-01")])
    assert r["effective_last_run_date"] == pd.Timestamp("2025-06-01")
    assert r["runs_newer_than_comprehensive"] is False


def test_runs_newer_wins_and_flags():
    r = _record(pd.Timestamp("2025-01-01"), runs_starts=[pd.Timestamp("2025-06-01")])
    assert r["effective_last_run_date"] == pd.Timestamp("2025-06-01")
    assert r["runs_newer_than_comprehensive"] is True


def test_runs_last_run_date_can_drive_effective():
    r = _record(pd.Timestamp("2025-01-01"), runs_last=[pd.Timestamp("2025-07-01")])
    assert r["effective_last_run_date"] == pd.Timestamp("2025-07-01")
    assert r["runs_newer_than_comprehensive"] is True


def test_comprehensive_blank_uses_runs():
    r = _record(pd.NaT, runs_starts=[pd.Timestamp("2025-06-01")])
    assert r["effective_last_run_date"] == pd.Timestamp("2025-06-01")
    assert r["runs_newer_than_comprehensive"] is True


def test_runs_blank_uses_comprehensive():
    r = _record(pd.Timestamp("2025-01-01"))
    assert r["effective_last_run_date"] == pd.Timestamp("2025-01-01")
    assert r["runs_newer_than_comprehensive"] is False


def test_both_blank_is_nat():
    r = _record(pd.NaT)
    assert pd.isna(r["effective_last_run_date"])
    assert r["runs_newer_than_comprehensive"] is False


def test_invalid_runs_timestamps_ignored():
    # An unparseable Runs timestamp must not crash and must not count as a date.
    r = _record(pd.Timestamp("2025-01-01"), runs_starts=["not-a-date"])
    assert r["effective_last_run_date"] == pd.Timestamp("2025-01-01")
    assert r["runs_exec_count"] == 1  # the row still counts toward run-count metrics
