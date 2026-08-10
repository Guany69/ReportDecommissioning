"""CLI entry point.

Usage:
  python -m report_cleanup.run --table1 metadata.xlsx --table2 execution.csv --out ./output
  python -m report_cleanup.run --table1 metadata.xlsx --table3-fields fields.xlsx --out ./output
  python -m report_cleanup.run --table1 metadata.xlsx --out ./output
"""
from __future__ import annotations

import argparse
import os
import sys

from .pipeline import run_pipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Report Decommissioning & Consolidation")
    p.add_argument("--table1", required=True, help="Metadata export (.csv/.xls/.xlsx)")
    p.add_argument("--table2", default=None, help="Execution/request export (.csv/.xls/.xlsx)")
    p.add_argument("--table3-fields", default=None, dest="table3_fields",
                   help="Field / data-dictionary / where-used export (.csv/.xls/.xlsx)")
    p.add_argument("--config", default=None, help="Path to config.yaml (defaults to bundled)")
    p.add_argument("--out", default="output", help="Output directory")
    p.add_argument("--sensitive", action="store_true",
                   help="Sensitive mode: mask person-identifying columns "
                        "(Owner, Created By, Last Run By, Requested By ...) in output.")
    args = p.parse_args(argv)

    # [SECURITY] Surface the flag as an env var so the export/preview layers
    # (which read it live) honor it for the rest of the process.
    if args.sensitive:
        os.environ["REPORT_CLEANUP_SENSITIVE_MODE"] = "1"

    res = run_pipeline(
        args.table1,
        args.table2,
        args.table3_fields,
        args.config,
        args.out,
    )

    n = len(res["records"])
    hard = sum(1 for r in res["records"] if r.get("is_hard_rule"))
    print(f"Run {res['run_id']} complete: {n} reports analyzed.")
    print(f"  Hard-rule deletes : {hard}")
    print(f"  Duplicate groups  : {len(res['groups'])}")
    print(f"  Join (composite/name-only/unmatched/ambiguous): "
          f"{res['diag']['composite']}/{res['diag']['name_only']}/"
          f"{res['diag']['unmatched']}/{res['diag']['ambiguous']}")

    fd = res.get("field_diag", {})
    print(f"  Field export mode : {fd.get('field_export_mode', 'n/a')}")
    if fd.get("field_export_mode") == "report_field_mapping":
        print(f"  Fields matched    : {fd.get('total_reports_with_fields', 0)}/{n} reports "
              f"({fd.get('field_join_match_rate', 0)}%)")
    if fd.get("metadata_only_pairs", 0):
        print(f"  Metadata-similar (no field evidence): {fd['metadata_only_pairs']} pairs")

    duplicate_scoring = res.get("duplicate_scoring", {})
    mode = duplicate_scoring.get("duplicate_scoring_mode", "weighted_baseline")
    status = duplicate_scoring.get("model_status", "disabled")
    version = duplicate_scoring.get("model_version")
    model_detail = f", model {version}" if version else ""
    print(f"  Duplicate scoring : {mode} ({status}{model_detail})")
    print(f"  Candidate pairs   : {duplicate_scoring.get('candidate_pair_count', 0)} "
          f"({duplicate_scoring.get('pairs_scored_by_ml', 0)} scored by ML)")

    for w in res["warnings"]:
        print(f"  WARN: {w}")
    print(f"  Excel  : {res['xlsx']}")
    print(f"  SQLite : {res['db']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
