"""Train the PyTorch duplicate classifier from a human-labeled pair CSV.

    python -m training.train_duplicate_model \
        --data data/training/candidate_pairs.csv \
        --output models/duplicate_model.pt

What this script guarantees:

* **No leakage.** The split is group-aware over connected components of report
  IDs (see `split.py`), so a report in training never reappears in validation or
  test. Zero-overlap is asserted, not assumed.
* **No accidental imbalance blindness.** Class counts are printed before the first
  epoch, and the positive class is up-weighted inside `BCEWithLogitsLoss` via
  `pos_weight = negatives / positives`. This reweights the loss rather than
  duplicating rows: oversampling would make the same handful of reviewed positives
  appear many times per epoch and invite memorization of those specific pairs.
* **No "last epoch wins".** The checkpoint with the best validation average
  precision is kept, and training stops early once validation stops improving.
* **No arbitrary 0.5 threshold.** The decision threshold is selected on the
  validation split under the documented precision-first criterion in
  `metrics.select_threshold`, then frozen into the artifact.

The held-out test split is NOT touched here. Use `evaluate_duplicate_model` after
this script, once the model and threshold are fixed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from report_cleanup.ml.artifact import build_metadata, save_artifact
from report_cleanup.ml.features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from report_cleanup.ml.model import DEFAULT_HIDDEN_SIZES, build_model

from .dataset import (LabeledPair, PairDataset, load_labeled_pairs,
                      require_baseline_fields, subset)
from .metrics import average_precision, evaluate, select_threshold
from .split import (DEFAULT_RATIOS, build_split_manifest, default_manifest_path,
                    group_aware_split, write_split_manifest)

DEFAULT_DATA = "data/training/candidate_pairs.csv"
DEFAULT_OUTPUT = "models/duplicate_model.pt"


def set_seeds(seed: int) -> None:
    """Make a run reproducible as far as is practical.

    Covers Python, numpy, and torch RNGs. Full bit-for-bit determinism across
    machines is not promised (BLAS kernels differ); same-machine reproducibility is.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """CUDA automatically under ``auto`` when available; CPU always works."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[warning] --device cuda requested but CUDA is unavailable; using CPU.",
                  file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def compute_pos_weight(pairs: Sequence[LabeledPair]) -> float:
    """negatives / positives, the standard BCEWithLogitsLoss imbalance correction.

    Returns 1.0 when either class is absent — there is nothing to rebalance, and a
    divide-by-zero here would silently produce inf weights.
    """
    positives = sum(1 for p in pairs if p.label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


@torch.inference_mode()
def predict_probabilities(model: nn.Module, pairs: Sequence[LabeledPair],
                          device: torch.device) -> list[float]:
    """Probabilities for a whole split in one pass."""
    if not pairs:
        return []
    model.eval()
    x = torch.tensor([p.features for p in pairs], dtype=torch.float32, device=device)
    return torch.sigmoid(model(x)).detach().cpu().tolist()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over the training split. Returns the mean minibatch loss."""
    model.train()
    total = 0.0
    seen = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * xb.size(0)
        seen += xb.size(0)
    return total / seen if seen else 0.0


def _print_class_distribution(name: str, pairs: Sequence[LabeledPair]) -> None:
    pos = sum(1 for p in pairs if p.label == 1)
    neg = len(pairs) - pos
    rate = (pos / len(pairs) * 100.0) if pairs else 0.0
    print(f"  {name:<11} {len(pairs):>6} examples   {pos:>5} positive   "
          f"{neg:>5} negative   {rate:5.1f}% positive")


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full training job and return the summary that was printed."""
    set_seeds(args.seed)
    device = resolve_device(args.device)

    loaded = load_labeled_pairs(args.data, unresolved_policy=args.unresolved_labels)
    require_baseline_fields(loaded.pairs)
    counts = loaded.counts()

    print("Labeled dataset")
    print(f"  file:                  {args.data}")
    print(f"  rows in file:          {counts['rows_in_file']}")
    print(f"  usable labels:         {counts['labeled_examples']}")
    print(f"  excluded (blank/unsure): {counts['excluded_unresolved']} "
          f"[policy: {args.unresolved_labels}]")
    print(f"  positive:              {counts['positive']}")
    print(f"  negative:              {counts['negative']}")
    print(f"  positive rate:         {counts['positive_rate'] * 100:.2f}%")
    print()

    if not loaded.pairs:
        raise SystemExit(
            "No usable labeled examples. Fill the 'label' column with 1 / 0 "
            "(see `python -m training.generate_pairs`) before training."
        )
    if counts["positive"] == 0 or counts["negative"] == 0:
        raise SystemExit(
            "Training needs both classes present. Found "
            f"{counts['positive']} positive and {counts['negative']} negative examples."
        )

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    split = group_aware_split(loaded.id_pairs(), ratios=ratios, seed=args.seed)
    train_pairs = subset(loaded.pairs, split.train)
    val_pairs = subset(loaded.pairs, split.val)
    test_pairs = subset(loaded.pairs, split.test)

    print("Group-aware split (connected components of report IDs; zero ID overlap verified)")
    _print_class_distribution("train", train_pairs)
    _print_class_distribution("validation", val_pairs)
    _print_class_distribution("test", test_pairs)
    print(f"  distinct report IDs per split: {split.id_counts()}")
    print()

    if not train_pairs:
        raise SystemExit("The split produced an empty training set — add more labeled pairs.")
    if not val_pairs:
        raise SystemExit(
            "The split produced an empty validation set. Model selection and threshold "
            "selection both need one; add more labeled pairs or widen --val-ratio."
        )

    # A single-class validation or test split makes the numbers computed from it
    # meaningless — precision is trivially 1.0 with no negatives, and recall is
    # undefined with no positives — so say so loudly rather than let a perfect-looking
    # threshold be mistaken for a validated one. This is common with group-aware
    # splitting when duplicate clusters are small: every pair inside one cluster can
    # share a label.
    degenerate: list[str] = []
    for name, pairs in (("validation", val_pairs), ("test", test_pairs)):
        if not pairs:
            continue
        positives = sum(1 for p in pairs if p.label == 1)
        if positives in (0, len(pairs)):
            only = "positive" if positives else "negative"
            degenerate.append(name)
            print(f"[WARNING] The {name} split is 100% {only} examples. Metrics and any "
                  f"threshold selected from it are NOT meaningful. Label more pairs so "
                  f"each split contains both classes.")
    if degenerate:
        print("[WARNING] This model is NOT validated. Do not enable it in production "
              f"on the strength of {'/'.join(degenerate)} numbers from this run.")
        print()
    if not test_pairs:
        raise SystemExit(
            "The split produced an empty held-out test set. Add more disconnected "
            "report-pair components before training."
        )
    if {pair.label for pair in train_pairs} != {0, 1}:
        raise SystemExit(
            "The training split must contain both classes. Add human-reviewed "
            "examples from more disconnected report components."
        )

    pos_weight = compute_pos_weight(train_pairs)
    print(f"Class imbalance: pos_weight = negatives/positives = {pos_weight:.3f} "
          "(applied inside BCEWithLogitsLoss)")
    print()

    model = build_model(
        input_size=FEATURE_COUNT,
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=min(args.batch_size, len(train_pairs)),
        shuffle=True,
        generator=generator,
    )

    val_labels = [p.label for p in val_pairs]
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(f"Training on {device} for up to {args.epochs} epochs "
          f"(early stopping patience {args.patience})")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        val_probs = predict_probabilities(model, val_pairs, device)
        # Average precision, not accuracy or loss: it is the metric that stays
        # informative when positives are rare.
        val_ap = average_precision(val_labels, val_probs)
        score = -1.0 if val_ap is None else val_ap
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_average_precision": None if val_ap is None else round(val_ap, 6)})

        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % args.log_every == 0 or improved or epoch == 1:
            marker = " *" if improved else ""
            ap_txt = "n/a" if val_ap is None else f"{val_ap:.4f}"
            print(f"  epoch {epoch:>4}  train_loss {train_loss:.5f}  val_AP {ap_txt}{marker}")

        if epochs_without_improvement >= args.patience:
            print(f"  early stopping at epoch {epoch} "
                  f"(no validation improvement for {args.patience} epochs)")
            break

    if best_state is None:
        raise SystemExit("Training produced no usable checkpoint.")

    # Restore the BEST validation checkpoint, never the final epoch's weights.
    model.load_state_dict(best_state)
    model.eval()
    print(f"Restored best checkpoint from epoch {best_epoch} (val AP {best_score:.4f})")
    print()

    # Threshold selection happens on VALIDATION only; test stays untouched here.
    val_probs = predict_probabilities(model, val_pairs, device)
    threshold, criterion_text = select_threshold(
        val_labels, val_probs, min_precision=args.min_precision)
    val_metrics = evaluate(val_labels, val_probs, threshold)

    print("Threshold selection (validation split)")
    print(f"  criterion: {criterion_text}")
    print(f"  threshold: {threshold:.4f}")
    print(f"  validation precision {val_metrics.precision:.4f}  "
          f"recall {val_metrics.recall:.4f}  F1 {val_metrics.f1:.4f}")
    print()

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = build_split_manifest(
        split,
        loaded.pairs,
        dataset_sha256=loaded.sha256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        seed=args.seed,
        ratios=ratios,
        unresolved_policy=args.unresolved_labels,
    )
    requested_manifest = (
        Path(args.split_manifest) if args.split_manifest
        else default_manifest_path(args.output)
    )
    manifest_path, manifest_sha256 = write_split_manifest(
        requested_manifest, manifest)

    metadata = build_metadata(
        model=model,
        model_version=args.model_version or f"duplicate-mlp-{trained_at[:10]}-seed{args.seed}",
        decision_threshold=threshold,
        threshold_criterion=criterion_text,
        trained_at=trained_at,
        seed=args.seed,
        hyperparameters={
            "epochs_requested": args.epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_sizes": list(args.hidden_sizes),
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "pos_weight": round(pos_weight, 6),
            "early_stopping_patience": args.patience,
            "min_delta": args.min_delta,
            "split_ratios": list(ratios),
            "min_precision": args.min_precision,
        },
        dataset_counts={
            **counts,
            "train": len(train_pairs),
            "validation": len(val_pairs),
            "test": len(test_pairs),
            "train_positive": sum(1 for p in train_pairs if p.label == 1),
            "validation_positive": sum(1 for p in val_pairs if p.label == 1),
            "test_positive": sum(1 for p in test_pairs if p.label == 1),
            "source_file": Path(args.data).name,
            # The exact bytes reviewed, so a metric can always be traced to a file.
            "source_sha256": loaded.sha256,
            "unresolved_label_policy": args.unresolved_labels,
        },
        evaluation_metrics={
            "validation": val_metrics.as_dict(),
            # Persisted so a later reader of the artifact learns the threshold came
            # from a split that could not actually validate it.
            "degenerate_single_class_splits": degenerate,
        },
        notes=args.notes,
    )
    metadata["training_data"] = {
        "dataset_sha256": loaded.sha256,
        "source_file": Path(args.data).name,
        "unresolved_label_policy": args.unresolved_labels,
    }
    metadata["split_manifest"] = {
        "file": manifest_path.name,
        "sha256": manifest_sha256,
        "version": manifest["manifest_version"],
        "strategy": manifest["split_strategy"],
    }

    out = save_artifact(args.output, model, metadata)
    print(f"Saved model artifact: {out}")
    print(f"  model_version: {metadata['model_version']}")
    print(f"  dataset_sha256: {loaded.sha256}")
    print(f"  split_manifest: {manifest_path} (sha256 {manifest_sha256})")
    print()
    print("The held-out test split has NOT been scored. Run:")
    print(f"  python -m training.evaluate_duplicate_model --data {args.data} --model {out}")

    summary = {
        "artifact": str(out),
        "model_version": metadata["model_version"],
        "decision_threshold": threshold,
        "threshold_criterion": criterion_text,
        "best_epoch": best_epoch,
        "dataset_counts": metadata["dataset_counts"],
        "validation_metrics": val_metrics.as_dict(),
        "history": history,
        "dataset_sha256": loaded.sha256,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest_sha256,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Training summary written to {args.summary_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m training.train_duplicate_model",
        description="Train the PyTorch duplicate classifier on human-labeled pairs.",
    )
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"Labeled pair CSV (default {DEFAULT_DATA}).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Artifact path (default {DEFAULT_OUTPUT}).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES),
                    help="Hidden layer widths (default: 32 16).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unresolved-labels", choices=("exclude", "reject"), default="exclude",
                    help="What to do with blank / 'unsure' labels: 'exclude' drops and "
                         "counts them (default), 'reject' refuses an incomplete review.")
    ap.add_argument("--patience", type=int, default=20,
                    help="Stop after this many epochs without validation improvement.")
    ap.add_argument("--min-delta", type=float, default=1e-5,
                    help="Minimum validation AP gain that counts as an improvement.")
    ap.add_argument("--min-precision", type=float, default=0.90,
                    help="Precision floor for threshold selection (default 0.90). False "
                         "positives push live reports toward decommissioning review.")
    ap.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    ap.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                    help="Automatically use CUDA when available; CPU is always supported.")
    ap.add_argument("--split-manifest", default=None,
                    help="Companion split JSON (default: <output>.split.json).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--model-version", default=None, help="Override the generated model version.")
    ap.add_argument("--notes", default="", help="Free text stored in the artifact metadata.")
    ap.add_argument("--summary-json", default=None, help="Also write the training summary here.")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
