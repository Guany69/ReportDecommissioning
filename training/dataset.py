"""Canonical reviewed-pair schema and PyTorch ``Dataset`` implementation.

The CSV stores both reviewer-friendly raw percentages and the exact normalized
``FEATURE_NAMES`` vector. Loading recomputes that vector from the raw signals and
requires an exact schema-version/order match, preventing a stale or hand-mutated
file from silently training on a different representation than inference.

Labels are ground truth, not heuristic output. Unambiguous positive/negative
tokens are accepted; blank/``unsure`` values follow an explicit loader policy.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
from torch.utils.data import Dataset

from report_cleanup.ml.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    RAW_SIGNAL_NAMES,
    build_feature_vector,
)

from . import (
    BASELINE_PREDICTION_COLUMN,
    BASELINE_RELATIONSHIP_COLUMN,
    BASELINE_SCORE_COLUMN,
    BASELINE_THRESHOLD_COLUMN,
    CANONICAL_PAIR_COLUMNS,
    LABEL_COLUMN,
    REPORT_A_ID_COLUMN,
    REPORT_B_ID_COLUMN,
)

UnresolvedPolicy = Literal["reject", "exclude"]
SIGNAL_PREFIX = "sig_"
SIGNAL_COLUMNS: tuple[str, ...] = tuple(SIGNAL_PREFIX + name for name in RAW_SIGNAL_NAMES)
SIGNAL_MISSING_COLUMNS: tuple[str, ...] = tuple(
    SIGNAL_PREFIX + name + "_missing" for name in RAW_SIGNAL_NAMES)
IDENTITY_COLUMNS: tuple[str, ...] = CANONICAL_PAIR_COLUMNS

# Human-review context. None of these columns is fed to the network.
_PAIR_CONTEXT_BASES = (
    "data_source", "report_type", "category", "report_tag", "worklet",
    "field_count", "field_extraction_status", "report_fields", "business_objects",
)
#: Per-report context fields, emitted twice with _a / _b suffixes.
CONTEXT_FIELDS: tuple[str, ...] = _PAIR_CONTEXT_BASES
CONTEXT_COLUMNS: tuple[str, ...] = (
    *(f"{name}_a" for name in _PAIR_CONTEXT_BASES),
    *(f"{name}_b" for name in _PAIR_CONTEXT_BASES),
    "candidate_metadata_similarity",
    BASELINE_SCORE_COLUMN,
    BASELINE_PREDICTION_COLUMN,
    BASELINE_THRESHOLD_COLUMN,
    BASELINE_RELATIONSHIP_COLUMN,
    "feature_schema_version",
    "source_table1_sha256",
    "source_table2_sha256",
    "source_table3_sha256",
    "generated_at_utc",
)
REVIEW_COLUMNS: tuple[str, ...] = (
    "review_status", "reviewer", "review_notes", "label_source",
)
PAIR_CSV_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS
    + CONTEXT_COLUMNS
    + SIGNAL_COLUMNS
    + SIGNAL_MISSING_COLUMNS
    + FEATURE_NAMES
    + (LABEL_COLUMN,)
    + REVIEW_COLUMNS
)

# Reviewers fill this column in a spreadsheet, where "1" readily becomes "1.0"
# and "yes" is a natural thing to type. Every token below is unambiguous; anything
# outside these three sets is rejected rather than guessed at.
_POSITIVE_LABELS = {"1", "1.0", "yes", "y", "true", "duplicate"}
_NEGATIVE_LABELS = {"0", "0.0", "no", "n", "false", "not_duplicate", "notduplicate"}
_UNRESOLVED_LABELS = {"", "unsure", "unresolved", "unknown", "skip", "?", "tbd", "n/a", "na"}


class DatasetError(ValueError):
    """A pair CSV does not satisfy the canonical, auditable schema."""


# Descriptive alias used by external callers and tests.
DatasetValidationError = DatasetError


@dataclass(frozen=True)
class LabeledPair:
    """One resolved, human-reviewed candidate pair."""

    report_uid_a: str
    report_uid_b: str
    features: list[float]
    label: int
    baseline_score: float | None = None  # normalized to 0..1 on load
    baseline_prediction: int | None = None
    baseline_threshold: float | None = None
    row_number: int = 0
    context: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def pair(self) -> tuple[str, str]:
        return self.report_uid_a, self.report_uid_b

    @property
    def report_a_id(self) -> str:
        return self.report_uid_a

    @property
    def report_b_id(self) -> str:
        return self.report_uid_b


# Alternative descriptive name without breaking the initially published API.
PairExample = LabeledPair


@dataclass(frozen=True)
class LoadedPairs:
    """Examples and immutable source facts used in artifact provenance."""

    pairs: list[LabeledPair]
    total_rows: int
    excluded_unresolved: int
    path: Path
    sha256: str

    @property
    def examples(self) -> list[LabeledPair]:
        return self.pairs

    @property
    def n_positive(self) -> int:
        return sum(pair.label for pair in self.pairs)

    @property
    def n_negative(self) -> int:
        return len(self.pairs) - self.n_positive

    @property
    def positive_count(self) -> int:
        return self.n_positive

    @property
    def negative_count(self) -> int:
        return self.n_negative

    @property
    def positive_rate(self) -> float:
        return self.n_positive / len(self.pairs) if self.pairs else 0.0

    def id_pairs(self) -> list[tuple[str, str]]:
        return [pair.pair for pair in self.pairs]

    def counts(self) -> dict[str, Any]:
        return {
            "rows_in_file": self.total_rows,
            "labeled_examples": len(self.pairs),
            "excluded_unresolved": self.excluded_unresolved,
            "positive": self.n_positive,
            "negative": self.n_negative,
            "positive_rate": round(self.positive_rate, 6),
        }


LoadedPairData = LoadedPairs


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 of the exact labeled CSV bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(value: Any) -> int | None:
    """Return 1, 0, or ``None`` for a deliberately unresolved row; reject all else.

    An unrecognized value raises rather than being coerced or silently dropped: it
    is a data-entry mistake, and this column is the only ground truth the system
    has. Coercing it would put a guess into the training set.
    """
    normalized = "" if value is None else str(value).strip().casefold()
    if normalized in _UNRESOLVED_LABELS:
        return None
    if normalized in _POSITIVE_LABELS:
        return 1
    if normalized in _NEGATIVE_LABELS:
        return 0
    raise DatasetError(
        f"Unrecognized label {value!r}. Use 1 (duplicate) or 0 (not duplicate); "
        "leave blank or write 'unsure' to exclude the row from training."
    )


def _parse_signal(value: Any, column: str, row_number: int) -> float | None:
    raw = "" if value is None else str(value).strip()
    if raw == "" or raw.casefold() in {"n/a", "na", "none", "null"}:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise DatasetError(
            f"Row {row_number}: {column!r} must be a percentage in 0..100 or blank."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise DatasetError(
            f"Row {row_number}: {column!r} must be finite and in 0..100; got {value!r}."
        )
    return parsed


def _parse_feature(value: Any, column: str, row_number: int) -> float:
    raw = "" if value is None else str(value).strip()
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise DatasetError(
            f"Row {row_number}: model feature {column!r} must be numeric."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise DatasetError(
            f"Row {row_number}: model feature {column!r} must be finite and in 0..1."
        )
    if column.endswith("_missing") and parsed not in (0.0, 1.0):
        raise DatasetError(
            f"Row {row_number}: missingness feature {column!r} must be 0 or 1."
        )
    return parsed


def _parse_missing(value: Any, column: str, row_number: int) -> int:
    raw = "" if value is None else str(value).strip()
    if raw not in {"0", "1"}:
        raise DatasetError(f"Row {row_number}: {column!r} must be exactly 0 or 1.")
    return int(raw)


def _parse_baseline_score(value: Any, row_number: int) -> float | None:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise DatasetError(
            f"Row {row_number}: {BASELINE_SCORE_COLUMN!r} must be numeric."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise DatasetError(
            f"Row {row_number}: {BASELINE_SCORE_COLUMN!r} must be in 0..100."
        )
    return parsed / 100.0


def _parse_baseline_prediction(value: Any, row_number: int) -> int | None:
    raw = "" if value is None else str(value).strip()
    if raw == "":
        return None
    if raw not in {"0", "1"}:
        raise DatasetError(
            f"Row {row_number}: {BASELINE_PREDICTION_COLUMN!r} must be exactly 0 or 1."
        )
    return int(raw)


def _parse_baseline_threshold(value: Any, row_number: int) -> float | None:
    """Reviewer CSV stores the configured deterministic threshold as 0..100."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise DatasetError(
            f"Row {row_number}: {BASELINE_THRESHOLD_COLUMN!r} must be numeric."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise DatasetError(
            f"Row {row_number}: {BASELINE_THRESHOLD_COLUMN!r} must be in 0..100."
        )
    return parsed / 100.0


# What a labeled file must contain to be trainable. Reviewer-only prose columns
# may be removed, but schema/version, raw availability, exact model vector, and
# baseline evidence are contractual and must survive spreadsheet editing.
REQUIRED_COLUMNS: tuple[str, ...] = (
    REPORT_A_ID_COLUMN,
    REPORT_B_ID_COLUMN,
    LABEL_COLUMN,
    "feature_schema_version",
    BASELINE_SCORE_COLUMN,
    BASELINE_PREDICTION_COLUMN,
    BASELINE_THRESHOLD_COLUMN,
    *SIGNAL_COLUMNS,
    *SIGNAL_MISSING_COLUMNS,
    *FEATURE_NAMES,
)


def _check_optional_cross_columns(
    row: dict[str, Any],
    raw_signals: dict[str, float | None],
    fieldnames: Sequence[str],
    row_number: int,
) -> None:
    """Validate the redundant audit columns when the reviewer kept them.

    `sig_*_missing` and the eighteen `FEATURE_NAMES` columns restate information
    already carried by the `sig_*` percentages. They exist so an auditor can see
    exactly what the network was handed. When present they must agree with the
    canonical transform — a disagreement means the file was hand-edited or was
    produced by a different feature schema, and training on it would be scoring a
    representation nobody reviewed.
    """
    for signal_name, signal_column, missing_column in zip(
        RAW_SIGNAL_NAMES, SIGNAL_COLUMNS, SIGNAL_MISSING_COLUMNS, strict=True
    ):
        if missing_column not in fieldnames or not str(row.get(missing_column) or "").strip():
            continue
        indicator = _parse_missing(row.get(missing_column), missing_column, row_number)
        if (raw_signals[signal_name] is None) != bool(indicator):
            raise DatasetError(
                f"Row {row_number}: {signal_column!r} and {missing_column!r} "
                "disagree about availability."
            )

    if not all(name in fieldnames for name in FEATURE_NAMES):
        return
    stored_raw = [str(row.get(name) or "").strip() for name in FEATURE_NAMES]
    if not all(stored_raw):
        return      # blank audit columns: nothing to cross-check against
    expected = build_feature_vector(raw_signals)
    for index, name in enumerate(FEATURE_NAMES):
        stored = _parse_feature(row.get(name), name, row_number)
        if abs(expected[index] - stored) > 1e-6:
            raise DatasetError(
                f"Row {row_number}: stored feature {name!r}={stored} does not match "
                f"the canonical transform ({expected[index]}). Regenerate the pair CSV."
            )


def load_labeled_pairs(
    path: str | Path,
    unresolved_policy: UnresolvedPolicy = "exclude",
) -> LoadedPairs:
    """Read a reviewer-labeled pair CSV into training examples.

    The model input is recomputed from the `sig_*` percentage columns rather than
    read from the stored feature columns, so there is exactly one implementation of
    the normalization and a stale CSV cannot feed a different representation into
    training than inference uses.

    ``unresolved_policy`` decides what a blank / 'unsure' label means:
    ``"exclude"`` (the default) drops the row and counts it, so the count can be
    reported and stored in the artifact; ``"reject"`` refuses the whole file, for
    callers who require a completed review.
    """
    if unresolved_policy not in {"reject", "exclude"}:
        raise ValueError("unresolved_policy must be 'reject' or 'exclude'.")
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"Labeled pair file not found: {source}")

    pairs: list[LabeledPair] = []
    total = 0
    excluded = 0
    seen_pairs: dict[tuple[str, str], int] = {}

    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DatasetError(f"{source} is empty — no header row.")
        fieldnames = list(reader.fieldnames)
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise DatasetError(
                f"{source} is missing required column(s): {', '.join(missing)}. "
                "Regenerate it with `python -m training.generate_pairs`."
            )
        feature_positions = [fieldnames.index(name) for name in FEATURE_NAMES]
        expected_positions = list(range(
            feature_positions[0], feature_positions[0] + len(FEATURE_NAMES)))
        if feature_positions != expected_positions:
            raise DatasetError(
                "Model feature columns must be contiguous and ordered exactly as "
                f"FEATURE_NAMES: {list(FEATURE_NAMES)}."
            )

        for row_number, row in enumerate(reader, start=2):   # row 1 is the header
            total += 1
            try:
                label = parse_label(row.get(LABEL_COLUMN))
            except DatasetError as exc:
                raise DatasetError(f"Row {row_number}: {exc}") from exc
            if label is None:
                if unresolved_policy == "reject":
                    raise DatasetError(
                        f"Row {row_number}: unresolved label. Complete the review, or "
                        "load with unresolved_policy='exclude'."
                    )
                excluded += 1
                continue

            uid_a = str(row.get(REPORT_A_ID_COLUMN) or "").strip()
            uid_b = str(row.get(REPORT_B_ID_COLUMN) or "").strip()
            if not uid_a or not uid_b:
                raise DatasetError(
                    f"Row {row_number}: report_uid_a and report_uid_b must both be non-blank.")
            if uid_a == uid_b:
                raise DatasetError(
                    f"Row {row_number}: report {uid_a!r} cannot be paired with itself.")
            pair_key = tuple(sorted((uid_a, uid_b)))
            if pair_key in seen_pairs:
                raise DatasetError(
                    f"Row {row_number}: duplicate unordered pair {pair_key}; first seen at "
                    f"row {seen_pairs[pair_key]}.")
            seen_pairs[pair_key] = row_number

            schema_version = str(row.get("feature_schema_version") or "").strip()
            if schema_version != FEATURE_SCHEMA_VERSION:
                raise DatasetError(
                    f"Row {row_number}: feature schema version {schema_version!r} does not "
                    f"match this build's {FEATURE_SCHEMA_VERSION!r}. Regenerate the pair CSV.")

            raw_signals = {
                name: _parse_signal(row.get(column), column, row_number)
                for name, column in zip(RAW_SIGNAL_NAMES, SIGNAL_COLUMNS, strict=True)
            }
            _check_optional_cross_columns(row, raw_signals, fieldnames, row_number)

            pairs.append(LabeledPair(
                report_uid_a=uid_a,
                report_uid_b=uid_b,
                features=build_feature_vector(raw_signals),
                label=label,
                baseline_score=_parse_baseline_score(
                    row.get(BASELINE_SCORE_COLUMN), row_number),
                baseline_prediction=_parse_baseline_prediction(
                    row.get(BASELINE_PREDICTION_COLUMN), row_number),
                baseline_threshold=_parse_baseline_threshold(
                    row.get(BASELINE_THRESHOLD_COLUMN), row_number),
                row_number=row_number,
                context={name: row.get(name, "") for name in CONTEXT_COLUMNS
                         if name in fieldnames},
            ))

    return LoadedPairs(
        pairs=pairs,
        total_rows=total,
        excluded_unresolved=excluded,
        path=source,
        sha256=dataset_sha256(source),
    )


class PairDataset(Dataset):
    """Float32 tensors suitable for ``BCEWithLogitsLoss``."""

    def __init__(self, pairs: Sequence[LabeledPair]) -> None:
        self.pairs = list(pairs)
        if self.pairs:
            self.x = torch.tensor(
                [pair.features for pair in self.pairs], dtype=torch.float32)
            self.y = torch.tensor(
                [float(pair.label) for pair in self.pairs], dtype=torch.float32)
        else:
            self.x = torch.empty((0, FEATURE_COUNT), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


DuplicatePairDataset = PairDataset


def subset(pairs: Sequence[LabeledPair], indices: Iterable[int]) -> list[LabeledPair]:
    return [pairs[index] for index in indices]


def require_baseline_fields(pairs: Sequence[LabeledPair]) -> None:
    """Require exact recorded baseline scores and verdicts for fair comparison."""
    missing = [
        pair.row_number for pair in pairs
        if (pair.baseline_score is None or pair.baseline_prediction is None
            or pair.baseline_threshold is None)
    ]
    if missing:
        raise DatasetError(
            "Exact baseline comparison requires populated baseline_similarity and "
            "baseline_prediction, and baseline_decision_threshold on CSV row(s): "
            + ", ".join(map(str, missing[:10]))
        )
