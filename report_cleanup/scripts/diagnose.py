"""Diagnose 'everything gets the same score'.

Shows (a) which source columns mapped to each scoring field, (b) how populated
each scoring column actually is after cleaning, and (c) the spread of Overall
Scores. If a scoring column maps but is ~0% populated, that is why every report
collapses onto the same score.

Usage (from the report_cleanup/ directory):
  python scripts/diagnose.py --table1 metadata.csv --table2 runs.csv --table3-fields fields.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make the project root importable when run as `python scripts/diagnose.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from report_cleanup import schema
from report_cleanup.clean import clean_table
from report_cleanup.config import load_config
from report_cleanup.io_readers import read_any
from report_cleanup.pipeline import run_pipeline

SCORING_DATE_NUM = ["last_run_date", "last_updated", "times_run", "shared", "worklet", "areas_used"]


def _coverage(series) -> float:
    n = max(len(series), 1)
    filled = series.map(lambda v: v is not None and not pd.isna(v) and str(v).strip() != "").sum()
    return round(100.0 * filled / n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table1", required=True)
    p.add_argument("--table2", default=None)
    p.add_argument("--table3-fields", dest="t3", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    raw = read_any(args.table1)
    print(f"\nComprehensive: {len(raw)} rows, {len(raw.columns)} columns")
    print("Raw headers:", list(raw.columns))

    m1 = schema.auto_map(list(raw.columns), cfg.aliases, schema.TABLE1_FIELDS)
    print("\n=== Scoring-field mapping + coverage (after cleaning) ===")
    inv = {src: key for key, src in m1.items()}
    cleaned = clean_table(raw[[c for c in raw.columns if c in inv]].rename(columns=inv),
                          schema.FIELD_TYPES, cfg.clean)
    for k in SCORING_DATE_NUM:
        src = m1.get(k, "*** UNMAPPED ***")
        cov = _coverage(cleaned[k]) if k in cleaned.columns else 0.0
        flag = "  <-- ALL BLANK (collapses the score!)" if (k in cleaned.columns and cov == 0.0) else ""
        print(f"  {k:14} <- {src:28} populated: {cov:5}%{flag}")

    print("\n=== Running full pipeline ===")
    res = run_pipeline(args.table1, args.table2, args.t3, args.config, out_dir="diagnose_out")
    scores = [r.get("overall_score") for r in res["records"]]
    dist = Counter(scores)
    print(f"Reports: {len(scores)}")
    print(f"Distinct Overall Scores: {len(dist)}")
    print("Score distribution (score: count):")
    for score, cnt in sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0])):
        print(f"  {score}: {cnt}")
    if len(dist) == 1:
        print("\n>>> Every report has the SAME score. Check the coverage table above:")
        print(">>> a scoring column showing 0% populated is the cause.")
    for w in res["warnings"]:
        print("WARN:", w)


if __name__ == "__main__":
    main()
