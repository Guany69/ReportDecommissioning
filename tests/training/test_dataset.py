"""Reviewer CSV parsing and PyTorch pair Dataset tests."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from report_cleanup.ml.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RAW_SIGNAL_NAMES,
    build_feature_vector,
)
from training import BASELINE_PREDICTION_COLUMN, BASELINE_THRESHOLD_COLUMN
from training.dataset import (
    LABEL_COLUMN,
    PAIR_CSV_COLUMNS,
    SIGNAL_MISSING_COLUMNS,
    SIGNAL_PREFIX,
    DatasetError,
    LabeledPair,
    PairDataset,
    load_labeled_pairs,
    parse_label,
    subset,
)


def _row(uid_a: str, uid_b: str, label: str, **overrides) -> dict[str, str]:
    row = {column: "" for column in PAIR_CSV_COLUMNS}
    row.update(
        {
            "report_uid_a": uid_a,
            "report_name_a": f"Report {uid_a}",
            "report_uid_b": uid_b,
            "report_name_b": f"Report {uid_b}",
            LABEL_COLUMN: label,
            "baseline_similarity": "75",
            BASELINE_PREDICTION_COLUMN: "1",
            BASELINE_THRESHOLD_COLUMN: "70",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
    )
    for index, name in enumerate(RAW_SIGNAL_NAMES):
        row[SIGNAL_PREFIX + name] = str(index * 10)
    row.update(overrides)

    # Populate the redundant audit columns exactly as the reviewer export does.
    # Invalid signal overrides intentionally keep a valid placeholder vector: the
    # loader must reject the malformed raw signal before it cross-checks the audit.
    raw: dict[str, float | None] = {}
    raw_is_valid = True
    for name, missing_column in zip(
        RAW_SIGNAL_NAMES, SIGNAL_MISSING_COLUMNS, strict=True
    ):
        value = str(row[SIGNAL_PREFIX + name]).strip()
        if not value:
            raw[name] = None
            row[missing_column] = "1"
            continue
        try:
            raw[name] = float(value)
        except ValueError:
            raw_is_valid = False
            raw[name] = 0.0
        row[missing_column] = "0"
    vector = build_feature_vector(raw) if raw_is_valid else [0.0] * FEATURE_COUNT
    row.update({name: str(value) for name, value in zip(FEATURE_NAMES, vector, strict=True)})
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], columns=PAIR_CSV_COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("value", [1, 1.0, "yes", "TRUE", "duplicate"])
def test_positive_labels_parse_as_one(value) -> None:
    assert parse_label(value) == 1


@pytest.mark.parametrize("value", [0, 0.0, "no", "FALSE", "not_duplicate"])
def test_negative_labels_parse_as_zero(value) -> None:
    assert parse_label(value) == 0


@pytest.mark.parametrize("value", [None, "", "  ", "unsure", "unknown", "?"])
def test_unresolved_labels_parse_explicitly_as_none(value) -> None:
    assert parse_label(value) is None


@pytest.mark.parametrize("value", [2, -1, "maybe", "duplicate-ish"])
def test_invalid_labels_are_rejected(value) -> None:
    with pytest.raises(DatasetError, match="Unrecognized label"):
        parse_label(value)


def test_loader_parses_labels_features_missingness_and_counts(tmp_path: Path) -> None:
    rows = [
        _row("A", "B", "1", sig_authorized_usage=""),
        _row("C", "D", "0", sig_data_source="0"),
        _row("E", "F", ""),
        _row("G", "H", "unsure"),
    ]
    loaded = load_labeled_pairs(_write_csv(tmp_path / "labeled.csv", rows))

    assert [pair.label for pair in loaded.pairs] == [1, 0]
    assert loaded.total_rows == 4
    assert loaded.excluded_unresolved == 2
    assert loaded.n_positive == 1
    assert loaded.n_negative == 1
    assert loaded.positive_rate == pytest.approx(0.5)
    assert loaded.id_pairs() == [("A", "B"), ("C", "D")]
    assert loaded.counts() == {
        "rows_in_file": 4,
        "labeled_examples": 2,
        "excluded_unresolved": 2,
        "positive": 1,
        "negative": 1,
        "positive_rate": 0.5,
    }
    first = dict(zip(FEATURE_NAMES, loaded.pairs[0].features, strict=True))
    assert first["field_jaccard"] == pytest.approx(0.0)
    assert first["field_jaccard_missing"] == 0.0
    assert first["authorized_usage"] == 0.0
    assert first["authorized_usage_missing"] == 1.0
    assert loaded.pairs[0].baseline_score == pytest.approx(0.75)


def test_loader_can_explicitly_reject_unresolved_labels(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "incomplete-review.csv",
        [_row("A", "B", "1"), _row("C", "D", "unsure")],
    )

    with pytest.raises(DatasetError, match=r"Row 3.*unresolved label"):
        load_labeled_pairs(path, unresolved_policy="reject")


def test_loader_rejects_invalid_label_with_source_row_number(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bad-label.csv", [_row("A", "B", "maybe")])

    with pytest.raises(DatasetError, match=r"Row 2.*Unrecognized label"):
        load_labeled_pairs(path)


def test_loader_rejects_invalid_signal_with_column_and_row(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "bad-signal.csv",
        [_row("A", "B", "1", sig_field_jaccard="not-a-number")],
    )

    with pytest.raises(DatasetError, match=r"Row 2.*sig_field_jaccard"):
        load_labeled_pairs(path)


@pytest.mark.parametrize("value", ["-0.1", "100.1", "nan", "inf"])
def test_loader_rejects_out_of_range_or_nonfinite_signal(
    tmp_path: Path, value: str
) -> None:
    path = _write_csv(
        tmp_path / "bad-range.csv",
        [_row("A", "B", "1", sig_field_jaccard=value)],
    )

    with pytest.raises(DatasetError, match=r"sig_field_jaccard.*0\.\.100"):
        load_labeled_pairs(path)


def test_loader_rejects_feature_schema_version_mismatch(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "stale-schema.csv",
        [_row("A", "B", "1", feature_schema_version="obsolete")],
    )

    with pytest.raises(DatasetError, match="feature schema version"):
        load_labeled_pairs(path)


def test_loader_rejects_audit_feature_that_disagrees_with_canonical_transform(
    tmp_path: Path,
) -> None:
    row = _row("A", "B", "1")
    row["field_jaccard"] = "0.999"
    path = _write_csv(tmp_path / "mutated-feature.csv", [row])

    with pytest.raises(DatasetError, match="does not match the canonical transform"):
        load_labeled_pairs(path)


def test_loader_rejects_missing_schema_columns(tmp_path: Path) -> None:
    columns = tuple(column for column in PAIR_CSV_COLUMNS if column != "sig_name")
    path = _write_csv(tmp_path / "missing-column.csv", [], columns=columns)

    with pytest.raises(DatasetError, match="sig_name"):
        load_labeled_pairs(path)


def test_loader_rejects_blank_pair_identifier(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "blank-id.csv", [_row("", "B", "1")])

    with pytest.raises(DatasetError, match="report_uid_a and report_uid_b"):
        load_labeled_pairs(path)


def test_pair_dataset_emits_float32_feature_and_target_tensors() -> None:
    examples = [
        LabeledPair("A", "B", [0.0] * FEATURE_COUNT, 0),
        LabeledPair("C", "D", [1.0] * FEATURE_COUNT, 1),
    ]

    dataset = PairDataset(examples)
    features, label = dataset[1]

    assert len(dataset) == 2
    assert dataset.x.shape == (2, FEATURE_COUNT)
    assert dataset.y.shape == (2,)
    assert dataset.x.dtype == torch.float32
    assert dataset.y.dtype == torch.float32
    assert features.dtype == torch.float32
    assert label.dtype == torch.float32
    assert label.item() == 1.0


def test_subset_preserves_requested_row_order() -> None:
    examples = [
        LabeledPair(str(index), str(index + 1), [0.0] * FEATURE_COUNT, index % 2)
        for index in range(4)
    ]

    selected = subset(examples, [3, 1])

    assert [pair.report_uid_a for pair in selected] == ["3", "1"]
