"""Classification metrics for an imbalanced binary problem.

Implemented on numpy rather than adding scikit-learn. These six functions are the
entire statistical surface this project needs; pulling in a large dependency (and
shipping it into the deployment environment) to get them is not a trade worth
making. Each is a direct, testable implementation of its textbook definition.

Raw accuracy is deliberately absent from `summarize`'s headline: with duplicates
rare among candidate pairs, predicting "never a duplicate" scores well on accuracy
and is useless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class ConfusionMatrix:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def as_dict(self) -> dict[str, int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class EvaluationResult:
    """Everything needed to judge one scorer at one threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None            # None when only one class is present
    average_precision: float | None  # PR-AUC; None when there are no positives
    confusion: ConfusionMatrix = field(
        default_factory=lambda: ConfusionMatrix(0, 0, 0, 0))
    n_examples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "precision": _r(self.precision),
            "recall": _r(self.recall),
            "f1": _r(self.f1),
            "roc_auc": _r(self.roc_auc),
            "pr_auc": _r(self.average_precision),
            "average_precision": _r(self.average_precision),
            "confusion_matrix": self.confusion.as_dict(),
            "n_examples": self.n_examples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)


def _as_arrays(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)
    if yt.shape != ys.shape:
        raise ValueError(f"y_true and y_score length mismatch: {yt.shape} vs {ys.shape}")
    if yt.size and not np.isin(yt, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if ys.size and (not np.isfinite(ys).all() or np.any((ys < 0.0) | (ys > 1.0))):
        raise ValueError("y_score must contain finite probabilities in 0..1.")
    return yt, ys


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionMatrix:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    return ConfusionMatrix(
        true_positives=int(np.sum((yt == 1) & (yp == 1))),
        false_positives=int(np.sum((yt == 0) & (yp == 1))),
        true_negatives=int(np.sum((yt == 0) & (yp == 0))),
        false_negatives=int(np.sum((yt == 1) & (yp == 0))),
    )


def precision_recall_f1(cm: ConfusionMatrix) -> tuple[float, float, float]:
    """Precision / recall / F1 from a confusion matrix.

    A denominator of zero yields 0.0 rather than NaN: "predicted no positives" is
    reported as zero precision, which is the conservative reading for a system
    whose false positives push live reports toward decommissioning review.
    """
    tp, fp, fn = cm.true_positives, cm.false_positives, cm.false_negatives
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, ties averaged.

    Returns None when only one class is present — AUC is undefined there, and
    reporting 0.5 or 1.0 would be a fabricated number.
    """
    yt, ys = _as_arrays(y_true, y_score)
    n_pos = int(yt.sum())
    n_neg = int(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(ys, kind="mergesort")
    sorted_scores = ys[order]
    ranks = np.empty(ys.size, dtype=np.float64)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Average rank (1-based) across the tied block.
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    rank_sum_pos = float(ranks[yt == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """PR-AUC as the step-wise average precision: sum over thresholds of
    (recall_k - recall_{k-1}) * precision_k, walking scores high to low.

    Returns None when there are no positives. Preferred over ROC-AUC as the model
    selection metric here because it is sensitive to performance on the rare class.
    """
    yt, ys = _as_arrays(y_true, y_score)
    n_pos = int(yt.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    ap = 0.0
    prev_recall = 0.0
    tp = 0
    for i in range(yt_sorted.size):
        tp += int(yt_sorted[i])
        # Only emit a point at the end of a tied score block: examples with equal
        # scores cannot be separated by any threshold.
        if i + 1 < ys_sorted.size and ys_sorted[i + 1] == ys_sorted[i]:
            continue
        precision = tp / (i + 1)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def evaluate(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    predictions: Sequence[int] | None = None,
) -> EvaluationResult:
    """Full metric bundle for one scorer at one decision threshold.

    ``predictions`` supports the deterministic baseline, whose guarded strong-name
    rule can flag a pair independently of its weighted-score threshold. Curves use
    the continuous score; the confusion matrix uses the baseline's actual verdict.
    """
    yt, ys = _as_arrays(y_true, y_score)
    if predictions is None:
        y_pred = (ys >= threshold).astype(np.int64)
    else:
        y_pred = np.asarray(predictions, dtype=np.int64)
        if y_pred.shape != yt.shape:
            raise ValueError(
                f"predictions and y_true length mismatch: {y_pred.shape} vs {yt.shape}"
            )
        if y_pred.size and not np.isin(y_pred, (0, 1)).all():
            raise ValueError("predictions must contain only 0 and 1.")
    cm = confusion_matrix(yt, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)
    n_pos = int(yt.sum())
    return EvaluationResult(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(yt, ys),
        average_precision=average_precision(yt, ys),
        confusion=cm,
        n_examples=int(yt.size),
        n_positive=n_pos,
        n_negative=int(yt.size - n_pos),
    )


def select_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    min_precision: float = 0.90,
) -> tuple[float, str]:
    """Choose a production decision threshold on the VALIDATION split.

    Criterion, and the reason for it: a false positive here nominates a live,
    legitimately-used report for consolidation or decommissioning review, which
    costs reviewer trust and can end with a real report being retired. A false
    negative merely leaves a duplicate undetected for another cycle. So the
    objective is **maximum F1 subject to precision >= min_precision**, with ties
    broken toward the stricter (higher) threshold. This documents the safety
    tradeoff explicitly instead of assuming 0.5 is right.

    If no candidate threshold reaches `min_precision`, the floor is dropped and the
    plain max-F1 threshold is returned — but the returned criterion string says so,
    and it is written into the artifact metadata, so nobody later mistakes it for a
    threshold that met the bar.

    Returns (threshold, human-readable criterion).
    """
    yt, ys = _as_arrays(y_true, y_score)
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be in 0..1.")
    if yt.size == 0:
        return 0.5, "default 0.5 (no validation examples available)"
    if int(yt.sum()) == 0:
        return 1.0, (
            "conservative 1.0 (validation split has no positive examples; "
            "a supervised production threshold cannot be selected)"
        )

    # Every distinct score is a candidate cut point, plus 0.5 so the conventional
    # default is always considered.
    candidates = sorted({float(s) for s in ys} | {0.5})

    best_constrained: tuple[float, float] | None = None   # (f1, threshold)
    best_overall: tuple[float, float] | None = None

    # Candidates ascend and ties use >=, so the LAST (highest, therefore strictest)
    # threshold wins. Two thresholds with equal F1 are not equally good here: the
    # stricter one sends fewer live reports to review.
    for threshold in candidates:
        result = evaluate(yt, ys, threshold)
        if best_overall is None or result.f1 >= best_overall[0]:
            best_overall = (result.f1, threshold)
        if result.precision >= min_precision:
            if best_constrained is None or result.f1 >= best_constrained[0]:
                best_constrained = (result.f1, threshold)

    if best_constrained is not None:
        threshold = best_constrained[1]
        chosen = evaluate(yt, ys, threshold)
        return threshold, (
            f"maximum validation F1 subject to precision >= {min_precision:.2f}, ties "
            f"broken toward the stricter threshold (selected precision="
            f"{chosen.precision:.4f}, recall={chosen.recall:.4f}, F1={chosen.f1:.4f})"
        )

    assert best_overall is not None
    threshold = best_overall[1]
    chosen = evaluate(yt, ys, threshold)
    return threshold, (
        f"maximum validation F1; NO threshold reached the precision floor of "
        f"{min_precision:.2f}, so the floor was not applied (selected precision="
        f"{chosen.precision:.4f}, recall={chosen.recall:.4f}, F1={chosen.f1:.4f}) — "
        "treat this model as not yet fit for production gating"
    )
