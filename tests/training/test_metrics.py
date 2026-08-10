"""Evaluation metrics and precision-first threshold selection tests."""
from __future__ import annotations

import pytest

from training.metrics import (
    average_precision,
    evaluate,
    roc_auc,
    select_threshold,
)


def test_evaluate_reports_imbalanced_binary_metrics_and_confusion_matrix() -> None:
    result = evaluate(
        y_true=[1, 0, 1, 0],
        y_score=[0.9, 0.8, 0.7, 0.1],
        threshold=0.75,
    )

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.roc_auc == pytest.approx(0.75)
    assert result.average_precision == pytest.approx(5.0 / 6.0)
    assert result.confusion.as_dict() == {
        "true_positives": 1,
        "false_positives": 1,
        "true_negatives": 1,
        "false_negatives": 1,
    }
    assert (result.n_examples, result.n_positive, result.n_negative) == (4, 2, 2)


def test_auc_metrics_handle_mathematically_undefined_single_class_cases() -> None:
    assert roc_auc([1, 1], [0.2, 0.8]) is None
    assert roc_auc([0, 0], [0.2, 0.8]) is None
    assert average_precision([0, 0], [0.2, 0.8]) is None
    assert average_precision([1, 1], [0.2, 0.8]) == pytest.approx(1.0)


def test_roc_auc_averages_tied_ranks() -> None:
    assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_threshold_selection_prioritizes_precision_floor() -> None:
    threshold, criterion = select_threshold(
        y_true=[1, 0, 1, 0],
        y_score=[0.9, 0.8, 0.7, 0.1],
        min_precision=0.9,
    )

    assert threshold == pytest.approx(0.9)
    selected = evaluate([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], threshold)
    assert selected.precision >= 0.9
    assert "precision" in criterion


def test_threshold_selection_discloses_when_precision_floor_is_unreachable() -> None:
    threshold, criterion = select_threshold(
        y_true=[0, 1],
        y_score=[0.9, 0.8],
        min_precision=0.95,
    )

    assert evaluate([0, 1], [0.9, 0.8], threshold).f1 == pytest.approx(2.0 / 3.0)
    assert "NO threshold reached" in criterion


def test_metric_inputs_reject_length_mismatch_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate([0, 1], [0.5], threshold=0.5)
    with pytest.raises(ValueError, match="only 0 and 1"):
        evaluate([0, 2], [0.2, 0.8], threshold=0.5)
