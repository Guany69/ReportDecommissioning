"""Evaluate the trained classifier against the deterministic baseline.

    python -m training.evaluate_duplicate_model \
        --data data/training/candidate_pairs.csv \
        --model models/duplicate_model.pt

The comparison is mandatory and it is the point of this script. The repository
already has a working weighted duplicate-similarity algorithm; a neural network is
only worth deploying if it beats that on the same held-out, human-reviewed pairs.
Both scorers see identical rows:

* **Baseline** — the weighted similarity recorded at export time, with the
  baseline's own verdict (which includes the guarded strong-name rule, so it is
  not merely `score >= possible`) driving its confusion matrix.
* **PyTorch** — the model's probability, thresholded at the value frozen into the
  artifact during training.

The artifact references a SHA-256-protected split manifest. Evaluation rejects a
changed/reordered dataset or manifest, then selects the exact rows frozen at
training time. Test data is scored here only after model/threshold selection.

This script reports what it measures. If the model does not beat the baseline, the
output says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from report_cleanup.ml.artifact import load_artifact
from report_cleanup.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from report_cleanup.ml.inference import DuplicatePredictor

from .dataset import LabeledPair, load_labeled_pairs, require_baseline_fields
from .metrics import EvaluationResult, evaluate
from .split import default_manifest_path, load_split_manifest

DEFAULT_DATA = "data/training/candidate_pairs.csv"
DEFAULT_MODEL = "models/duplicate_model.pt"

# The baseline flags a pair at/above duplicate_thresholds.possible. Used only when
# a row predates the baseline_prediction column.
FALLBACK_BASELINE_THRESHOLD = 0.70


def baseline_scores_and_predictions(
    pairs: Sequence[LabeledPair],
    fallback_threshold: float = FALLBACK_BASELINE_THRESHOLD,
) -> tuple[list[float], list[int], int]:
    """(scores 0..1, verdicts 0/1, rows missing a recorded baseline score)."""
    scores: list[float] = []
    predictions: list[int] = []
    missing = 0
    for p in pairs:
        if p.baseline_score is None:
            missing += 1
            scores.append(0.0)
        else:
            scores.append(min(1.0, max(0.0, p.baseline_score)))
        if p.baseline_prediction is not None:
            predictions.append(int(p.baseline_prediction))
        else:
            predictions.append(1 if scores[-1] >= fallback_threshold else 0)
    return scores, predictions, missing


def _fmt(v: float | None) -> str:
    return "   n/a" if v is None else f"{v:6.4f}"


def print_comparison(split_name: str, baseline: EvaluationResult,
                     model: EvaluationResult) -> None:
    print(f"--- {split_name} split: {model.n_examples} pairs "
          f"({model.n_positive} positive, {model.n_negative} negative) ---")
    print(f"{'metric':<20} {'baseline':>10} {'pytorch':>10}   delta")
    rows = (
        ("precision", baseline.precision, model.precision),
        ("recall", baseline.recall, model.recall),
        ("f1", baseline.f1, model.f1),
        ("roc_auc", baseline.roc_auc, model.roc_auc),
        ("pr_auc", baseline.average_precision, model.average_precision),
    )
    for name, b, m in rows:
        delta = "     n/a" if (b is None or m is None) else f"{m - b:+8.4f}"
        print(f"{name:<20} {_fmt(b):>10} {_fmt(m):>10}  {delta}")
    print(f"{'threshold':<20} {baseline.threshold:>10.4f} {model.threshold:>10.4f}")
    print()
    print(f"{'confusion':<20} {'baseline':>10} {'pytorch':>10}")
    for key in ("true_positives", "false_positives", "true_negatives", "false_negatives"):
        print(f"{key:<20} {baseline.confusion.as_dict()[key]:>10} "
              f"{model.confusion.as_dict()[key]:>10}")
    print()


def verdict(baseline: EvaluationResult, model: EvaluationResult) -> str:
    """An honest one-line summary. Never claims a win the numbers do not show."""
    if model.n_positive == 0:
        return ("INCONCLUSIVE — the held-out split contains no positive examples, so "
                "precision, recall, F1 and PR-AUC cannot distinguish the two scorers.")

    better_f1 = model.f1 > baseline.f1
    better_precision = model.precision >= baseline.precision
    have_ap = model.average_precision is not None and baseline.average_precision is not None
    better_ap = have_ap and model.average_precision > baseline.average_precision

    if better_f1 and better_ap and better_precision:
        return ("PyTorch outperforms the baseline on F1, PR-AUC, and precision on this "
                "held-out split.")
    if better_f1 or better_ap:
        return ("MIXED — PyTorch wins on some metrics and not others. Read the table "
                "above before deploying; do not enable it on the strength of one number.")
    return ("The PyTorch model does NOT outperform the deterministic weighted baseline "
            "on this held-out split. Keep ml_duplicate.enabled false.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_artifact(args.model)
    metadata = artifact.metadata
    training_data = metadata.get("training_data") or {}
    expected_dataset_sha = str(
        training_data.get("dataset_sha256")
        or (metadata.get("dataset_counts") or {}).get("source_sha256")
        or ""
    )
    if not expected_dataset_sha:
        raise SystemExit(
            "Artifact has no training dataset SHA-256; it predates leakage-safe "
            "evaluation and must be retrained."
        )
    unresolved_policy = str(
        training_data.get("unresolved_label_policy")
        or (metadata.get("dataset_counts") or {}).get("unresolved_label_policy")
        or "reject"
    )
    loaded = load_labeled_pairs(args.data, unresolved_policy=unresolved_policy)
    if loaded.sha256 != expected_dataset_sha:
        raise SystemExit(
            "Labeled dataset SHA-256 does not match the exact file used for training.\n"
            f"  expected: {expected_dataset_sha}\n  actual:   {loaded.sha256}\n"
            "Evaluation refused: changed labels, features, or row order would invalidate "
            "the held-out comparison."
        )

    manifest_metadata = metadata.get("split_manifest") or {}
    manifest_path = (
        Path(args.split_manifest) if args.split_manifest
        else artifact.path.parent / str(
            manifest_metadata.get("file") or default_manifest_path(artifact.path).name)
    )
    manifest, manifest_sha = load_split_manifest(manifest_path)
    expected_manifest_sha = str(manifest_metadata.get("sha256") or "")
    if not expected_manifest_sha or manifest_sha != expected_manifest_sha:
        raise SystemExit(
            "Split manifest SHA-256 does not match the model artifact. Evaluation refused.\n"
            f"  expected: {expected_manifest_sha or '<missing>'}\n"
            f"  actual:   {manifest_sha}"
        )
    if manifest.get("dataset_sha256") != loaded.sha256:
        raise SystemExit("Split manifest refers to a different labeled dataset SHA-256.")
    if (
        str(manifest.get("feature_schema_version")) != FEATURE_SCHEMA_VERSION
        or list(manifest.get("feature_names") or []) != list(FEATURE_NAMES)
    ):
        raise SystemExit("Split manifest feature schema is incompatible with this build.")

    split_entries = (manifest.get("splits") or {}).get(args.split)
    if not isinstance(split_entries, list):
        raise SystemExit(f"Split manifest has no valid {args.split!r} split.")
    pairs: list[LabeledPair] = []
    for entry in split_entries:
        try:
            index = int(entry["loaded_index"])
            pair = loaded.pairs[index]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid pair entry in split manifest: {entry!r}") from exc
        expected_key = (
            int(entry.get("csv_row_number", -1)),
            str(entry.get("report_uid_a", "")),
            str(entry.get("report_uid_b", "")),
        )
        actual_key = (pair.row_number, pair.report_uid_a, pair.report_uid_b)
        if expected_key != actual_key:
            raise SystemExit(
                "Split manifest row identity does not match the labeled dataset; "
                f"expected {expected_key}, found {actual_key}."
            )
        pairs.append(pair)
    require_baseline_fields(pairs)

    print("Evaluation")
    print(f"  model:            {artifact.path}")
    print(f"  model_version:    {artifact.model_version}")
    print(f"  trained_at:       {metadata.get('trained_at')}")
    print(f"  feature schema:   v{metadata.get('feature_schema_version')} "
          f"({len(artifact.feature_names)} features)")
    print(f"  decision threshold: {artifact.threshold:.4f}")
    print(f"  threshold criterion: {metadata.get('threshold_criterion')}")
    print(f"  labeled file:     {args.data}")
    print(f"  dataset SHA-256:  {loaded.sha256}")
    print(f"  split manifest:   {manifest_path} (verified SHA-256)")
    print(f"  split evaluated:  {args.split} (frozen at training time)")
    print()

    if not pairs:
        raise SystemExit(
            f"The frozen {args.split} split is empty. With few "
            "labeled components, a group-aware split can leave a split unpopulated — "
            "label more pairs rather than reducing the split ratios."
        )

    predictor = DuplicatePredictor(artifact, batch_size=args.batch_size)
    model_scores = predictor.predict([p.features for p in pairs])
    labels = [p.label for p in pairs]

    base_scores, base_predictions, _missing_baseline = baseline_scores_and_predictions(pairs)
    baseline_thresholds = {
        pair.baseline_threshold for pair in pairs if pair.baseline_threshold is not None
    }
    if len(baseline_thresholds) != 1:
        raise SystemExit(
            "Held-out rows must contain one consistent baseline_decision_threshold."
        )
    baseline_threshold = baseline_thresholds.pop()

    baseline_result = evaluate(labels, base_scores, baseline_threshold,
                               predictions=base_predictions)
    model_result = evaluate(labels, model_scores, predictor.threshold)

    print_comparison(args.split, baseline_result, model_result)
    print(verdict(baseline_result, model_result))
    print()
    print("Note: these metrics are only as trustworthy as the human labels behind "
          "them. A model trained on synthetic or heuristic-derived labels tells you "
          "nothing about real duplicate detection.")

    summary = {
        "model": str(artifact.path),
        "model_version": artifact.model_version,
        "split": args.split,
        "dataset_sha256": loaded.sha256,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest_sha,
        "seed": manifest.get("random_seed"),
        "split_ratios": manifest.get("split_ratios"),
        "baseline": baseline_result.as_dict(),
        "pytorch": model_result.as_dict(),
        "verdict": verdict(baseline_result, model_result),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m training.evaluate_duplicate_model",
        description="Compare the PyTorch duplicate classifier against the weighted baseline.",
    )
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"Labeled pair CSV (default {DEFAULT_DATA}).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Artifact (default {DEFAULT_MODEL}).")
    ap.add_argument("--split", choices=("test", "validation", "train"), default="test",
                    help="Which split to score. Default 'test' — use it only after "
                         "model and threshold selection are complete.")
    ap.add_argument("--split-manifest", default=None,
                    help="Split manifest path (defaults to the artifact companion).")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--output-json", default=None, help="Also write the summary here.")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    torch.manual_seed(0)
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
