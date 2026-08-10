"""Small synthetic end-to-end smoke test for training and held-out evaluation.

The labels here exist only to prove the local machinery works. No assertion treats
the resulting model quality as evidence about real Workday duplicate reports.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from report_cleanup.ml.artifact import load_artifact
from report_cleanup.ml.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RAW_SIGNAL_NAMES,
    build_feature_vector,
)
from report_cleanup.ml.inference import DuplicatePredictor
from training import (
    BASELINE_PREDICTION_COLUMN,
    BASELINE_SCORE_COLUMN,
    BASELINE_THRESHOLD_COLUMN,
    LABEL_COLUMN,
)
from training.dataset import (
    LabeledPair,
    PAIR_CSV_COLUMNS,
    SIGNAL_MISSING_COLUMNS,
    SIGNAL_PREFIX,
)
from training.evaluate_duplicate_model import (
    baseline_scores_and_predictions,
    build_parser as build_evaluation_parser,
    run as evaluate_model,
    verdict,
)
from training.metrics import evaluate
from training.split import group_aware_split
from training.train_duplicate_model import (
    build_parser as build_training_parser,
    compute_pos_weight,
    train,
)


def _write_balanced_synthetic_pairs(path: Path, *, count: int, seed: int) -> Path:
    id_pairs = [(f"A-{index}", f"B-{index}") for index in range(count)]
    split = group_aware_split(id_pairs, seed=seed)

    # Give every split both classes. This is test scaffolding, not generated
    # training evidence: the file lives only inside pytest's temporary directory.
    labels: dict[int, int] = {}
    for indices in (split.train, split.val, split.test):
        for position, row_index in enumerate(indices):
            labels[row_index] = position % 2

    rows: list[dict[str, str | int]] = []
    for index, (uid_a, uid_b) in enumerate(id_pairs):
        label = labels[index]
        row: dict[str, str | int] = {column: "" for column in PAIR_CSV_COLUMNS}
        row.update(
            {
                "report_uid_a": uid_a,
                "report_name_a": f"Synthetic A {index}",
                "report_uid_b": uid_b,
                "report_name_b": f"Synthetic B {index}",
                LABEL_COLUMN: label,
                BASELINE_SCORE_COLUMN: 95 if label else 5,
                BASELINE_PREDICTION_COLUMN: label,
                BASELINE_THRESHOLD_COLUMN: 70,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "reviewer": "pytest synthetic fixture",
                "review_notes": "Not real training evidence",
            }
        )
        signal = "95" if label else "5"
        raw_signals = {}
        for raw_name, missing_column in zip(
            RAW_SIGNAL_NAMES, SIGNAL_MISSING_COLUMNS, strict=True
        ):
            row[SIGNAL_PREFIX + raw_name] = signal
            row[missing_column] = "0"
            raw_signals[raw_name] = float(signal)
        row.update(dict(zip(
            FEATURE_NAMES, build_feature_vector(raw_signals), strict=True
        )))
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_positive_class_weight_uses_negative_to_positive_ratio() -> None:
    pairs = [
        LabeledPair(str(index), str(index + 1), [0.0] * len(FEATURE_NAMES), label)
        for index, label in enumerate([1, 0, 0, 0])
    ]

    assert compute_pos_weight(pairs) == pytest.approx(3.0)
    assert compute_pos_weight(pairs[:1]) == pytest.approx(1.0)


def test_baseline_evaluation_uses_recorded_verdict_not_only_score_threshold() -> None:
    pairs = [
        LabeledPair(
            "A",
            "B",
            [0.0] * len(FEATURE_NAMES),
            1,
            baseline_score=0.2,
            baseline_prediction=1,
        ),
        LabeledPair(
            "C",
            "D",
            [0.0] * len(FEATURE_NAMES),
            0,
            baseline_score=None,
            baseline_prediction=None,
        ),
    ]

    scores, predictions, missing = baseline_scores_and_predictions(pairs)

    assert scores == [0.2, 0.0]
    assert predictions == [1, 0]
    assert missing == 1


def test_evaluation_does_not_claim_model_superiority_when_metrics_do_not_show_it() -> None:
    labels = [1, 0, 1, 0]
    baseline = evaluate(labels, [0.99, 0.01, 0.98, 0.02], threshold=0.5)
    model = evaluate(labels, [0.6, 0.7, 0.4, 0.3], threshold=0.5)

    assert "does NOT outperform" in verdict(baseline, model)


def test_synthetic_training_artifact_inference_and_evaluation_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed = 19
    data_path = _write_balanced_synthetic_pairs(
        tmp_path / "synthetic_labeled_pairs.csv", count=20, seed=seed
    )
    artifact_path = tmp_path / "synthetic_duplicate_model.pt"
    summary_path = tmp_path / "synthetic_training_summary.json"

    train_args = build_training_parser().parse_args(
        [
            "--data", str(data_path),
            "--output", str(artifact_path),
            "--summary-json", str(summary_path),
            "--epochs", "4",
            "--batch-size", "4",
            "--patience", "2",
            "--learning-rate", "0.01",
            "--min-precision", "0.5",
            "--seed", str(seed),
            "--device", "cpu",
            "--model-version", "synthetic-smoke-only",
            "--notes", "Synthetic pytest smoke model; not production validated.",
            "--log-every", "10",
        ]
    )
    training_summary = train(train_args)

    assert artifact_path.exists()
    assert summary_path.exists()
    assert training_summary["model_version"] == "synthetic-smoke-only"
    assert 1 <= training_summary["best_epoch"] <= 4
    assert 0.0 <= training_summary["decision_threshold"] <= 1.0
    assert training_summary["dataset_counts"]["labeled_examples"] == 20

    artifact = load_artifact(artifact_path)
    assert artifact.model_version == "synthetic-smoke-only"
    assert artifact.feature_names == list(FEATURE_NAMES)
    assert artifact.metadata["notes"].startswith("Synthetic pytest")
    assert set(artifact.metadata["evaluation_metrics"]) == {
        "validation",
        "degenerate_single_class_splits",
    }
    predictor = DuplicatePredictor(artifact, batch_size=2, device="cpu")
    probabilities = predictor.predict(
        [[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)]
    )
    assert len(probabilities) == 2
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    evaluation_args = build_evaluation_parser().parse_args(
        [
            "--data", str(data_path),
            "--model", str(artifact_path),
            "--split", "test",
            "--batch-size", "2",
        ]
    )
    evaluation_summary = evaluate_model(evaluation_args)

    assert evaluation_summary["split"] == "test"
    assert evaluation_summary["baseline"]["n_examples"] > 0
    assert (
        evaluation_summary["baseline"]["n_examples"]
        == evaluation_summary["pytorch"]["n_examples"]
    )
    assert evaluation_summary["baseline"]["n_positive"] > 0
    assert evaluation_summary["baseline"]["n_negative"] > 0
    assert "verdict" in evaluation_summary
    output = capsys.readouterr().out
    assert "held-out" in output
    assert "only as trustworthy as the human labels" in output
