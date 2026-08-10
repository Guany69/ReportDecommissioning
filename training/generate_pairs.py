"""Export deterministic duplicate candidates for human labeling.

This command intentionally stops before any supervised training. It runs the
same local Workday ingestion, cleaning, field attachment, candidate blocking,
feature extraction, and weighted-baseline code used by production, then writes a
CSV whose ``label`` cells are blank. It never derives labels from the heuristic.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from report_cleanup.clean import text
from report_cleanup.config import load_config
from report_cleanup.dedup import candidate_similarity_score, generate_candidate_pairs
from report_cleanup.duplicate_similarity import compute_duplicate_similarity
from report_cleanup.io_readers import file_sha256
from report_cleanup.ml.features import (FEATURE_NAMES, FEATURE_SCHEMA_VERSION,
                                        RAW_SIGNAL_NAMES, build_feature_vector,
                                        extract_raw_signals)
from report_cleanup.pipeline import prepare_records
from report_cleanup.security import excel_safe, secure_mkdir

from . import (BASELINE_PREDICTION_COLUMN, BASELINE_RELATIONSHIP_COLUMN,
               BASELINE_SCORE_COLUMN, BASELINE_THRESHOLD_COLUMN, LABEL_COLUMN, REPORT_A_ID_COLUMN,
               REPORT_A_NAME_COLUMN, REPORT_B_ID_COLUMN, REPORT_B_NAME_COLUMN)
from .dataset import (CONTEXT_FIELDS, PAIR_CSV_COLUMNS, SIGNAL_COLUMNS,
                      SIGNAL_MISSING_COLUMNS)

#: The reviewer-CSV header is owned by `dataset.py` so the writer here and the
#: reader there can never disagree about column names or order.
OUTPUT_COLUMNS = PAIR_CSV_COLUMNS

def prepare_candidate_records(
    table1_path: str | Path,
    table2_path: str | Path | None,
    table3_fields_path: str | Path | None,
    config_path: str | Path | None,
) -> tuple[list[dict], Any, list[str]]:
    """Run the production preprocessing that pair features depend on.

    Delegates to `pipeline.prepare_records`, which is the same ingest -> validate
    -> clean -> field-rollup -> join -> recurrence -> attach-fields sequence
    `run_pipeline` uses. Reusing it (rather than re-implementing those steps here)
    is what guarantees the labeled pairs are extracted from records identical to
    the ones production scores. It covers only that read-only prefix: scoring,
    SQLite persistence, and the Excel export all live later in `run_pipeline`, so
    generating pairs never writes a run artifact.
    """
    cfg = load_config(config_path)
    prepared = prepare_records(table1_path, table2_path, table3_fields_path, cfg)
    return prepared.records, cfg, list(prepared.warnings)


def _context_value(record: dict, name: str) -> Any:
    if name == "business_objects":
        return "; ".join(sorted(map(str, record.get("business_objects_set") or set())))
    value = record.get(name)
    if isinstance(value, (set, list, tuple)):
        return "; ".join(sorted(map(str, value)))
    return text(value)


def _fmt_signal(value: float | None) -> str:
    """Percent to one decimal, or empty when the signal was unavailable.

    The empty-vs-"0.0" distinction has to survive all the way to the CSV: it is
    what `dataset.load_labeled_pairs` turns back into a missingness indicator, and
    it is what tells a reviewer "these reports share no data source" apart from
    "we do not know either report's data source".
    """
    return "" if value is None else f"{float(value):.1f}"


def build_pair_rows(
    records: Sequence[dict],
    cfg: Any,
    *,
    limit: int | None = None,
    generated_at: str = "",
    source_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build reviewer rows from the exact deterministic candidate set.

    Pairs come from `dedup.generate_candidate_pairs` — the same inverted-index
    blocking the runtime uses — so the labeled distribution matches what the model
    is actually asked to score. Sampling all pairs instead would train on a
    population the model never sees in production.

    Pairs are emitted in sorted index order, so a given input always produces the
    same rows in the same order (and `--limit` always takes the same prefix).
    """
    hashes = source_hashes or {}
    rows: list[dict[str, Any]] = []

    for i, j in sorted(generate_candidate_pairs(list(records), cfg)):
        if limit is not None and len(rows) >= limit:
            break
        report_a, report_b = records[i], records[j]
        # Round ONCE, before building the vector. The CSV publishes signals to one
        # decimal place, so deriving the feature columns from the unrounded values
        # would make the file internally inconsistent: the loader recomputes from
        # the published percentages and would find a mismatch it cannot explain.
        raw = {
            name: (None if value is None else round(float(value), 1))
            for name, value in extract_raw_signals(report_a, report_b, cfg).items()
        }
        vector = build_feature_vector(raw)
        # Default settings preserve the weighted baseline's exact production
        # behavior, including its correctness-preserving ceiling short-circuit.
        baseline = compute_duplicate_similarity(report_a, report_b, cfg)

        row: dict[str, Any] = {
            REPORT_A_ID_COLUMN: report_a["report_uid"],
            REPORT_A_NAME_COLUMN: excel_safe(text(report_a.get("report_name"))),
            REPORT_B_ID_COLUMN: report_b["report_uid"],
            REPORT_B_NAME_COLUMN: excel_safe(text(report_b.get("report_name"))),
            "candidate_metadata_similarity": candidate_similarity_score(
                report_a, report_b, cfg),
            BASELINE_SCORE_COLUMN: baseline.overall,
            BASELINE_PREDICTION_COLUMN: int(baseline.potential_duplicate),
            BASELINE_THRESHOLD_COLUMN: float(
                cfg.duplicate_thresholds.get("possible", 70)),
            BASELINE_RELATIONSHIP_COLUMN: baseline.relationship,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_table1_sha256": hashes.get("table1", ""),
            "source_table2_sha256": hashes.get("table2", ""),
            "source_table3_sha256": hashes.get("table3", ""),
            "generated_at_utc": generated_at,
            # Never populate this from the baseline. It belongs to a reviewer.
            LABEL_COLUMN: "",
            "review_status": "unreviewed",
            "reviewer": "",
            "review_notes": "",
            "label_source": "",
        }
        for context_name in CONTEXT_FIELDS:
            row[f"{context_name}_a"] = excel_safe(_context_value(report_a, context_name))
            row[f"{context_name}_b"] = excel_safe(_context_value(report_b, context_name))
        for signal_name, signal_column, missing_column in zip(
            RAW_SIGNAL_NAMES, SIGNAL_COLUMNS, SIGNAL_MISSING_COLUMNS, strict=True
        ):
            row[signal_column] = _fmt_signal(raw[signal_name])
            row[missing_column] = int(raw[signal_name] is None)
        # The exact 18 numbers the network would receive, for audit.
        row.update(dict(zip(FEATURE_NAMES, vector, strict=True)))
        rows.append(row)
    return rows


def write_pair_csv(
    rows: Sequence[dict[str, Any]], output: str | Path, *, overwrite: bool = False
) -> Path:
    """Write rows in the canonical column order to a 0600 file.

    Refuses to clobber an existing file unless `overwrite` is set: that file may
    hold hours of human review work, and regenerating over it is unrecoverable.
    """
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing labeling file {destination}. "
            "Pass --overwrite only if any review work there is expendable."
        )
    secure_mkdir(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # Candidate pairs carry report names and metadata — as sensitive as any other
    # engine output, so the file is owner-only rather than default 0644.
    if os.name == "posix":
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic duplicate candidates with blank human-review labels.")
    parser.add_argument("--table1", required=True, help="Workday report metadata (.csv/.xls/.xlsx)")
    parser.add_argument("--table2", help="Optional execution/request export (.csv/.xls/.xlsx)")
    parser.add_argument(
        "--table3-fields", dest="table3_fields",
        help="Optional field/where-used export (.csv/.xls/.xlsx)")
    parser.add_argument("--config", help="Config YAML (defaults to repository config.yaml)")
    parser.add_argument(
        "--output", required=True,
        help="Reviewer-ready CSV to create (for example data/training/candidate_pairs.csv)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of exported pairs (useful for a first labeling batch)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output file; use carefully because it may contain review work")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        records, cfg, warnings = prepare_candidate_records(
            args.table1, args.table2, args.table3_fields, args.config)
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_hashes = {
            "table1": file_sha256(args.table1),
            "table2": file_sha256(args.table2) if args.table2 else "",
            "table3": file_sha256(args.table3_fields) if args.table3_fields else "",
        }
        rows = build_pair_rows(
            records, cfg, limit=args.limit,
            generated_at=generated_at, source_hashes=source_hashes)
        destination = write_pair_csv(rows, args.output, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total = len(sorted(generate_candidate_pairs(list(records), cfg)))
    print(f"Wrote {len(rows):,} deterministic candidate pairs to {destination}.")
    if args.limit is not None and total > len(rows):
        # Never let a cap read as full coverage.
        print(f"WARNING: --limit {args.limit} capped the export; "
              f"{total - len(rows):,} of {total:,} candidate pairs were NOT written.")
    print("Labels were left blank; no heuristic or synthetic labels were generated.")
    print("Fill 'label' with 1 (duplicate) or 0 (not a duplicate). Blank/'unsure' "
          "rows require the explicit training option --unresolved-labels exclude.")
    print(f"Do NOT copy {BASELINE_PREDICTION_COLUMN} into label — that trains the model "
          "to imitate the heuristic and invalidates the baseline comparison.")
    if not rows:
        print("WARNING: no candidate pairs were produced from the configured blocking stage.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
