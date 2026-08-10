"""Reviewer-ready candidate export tests; labels must never be fabricated."""
from __future__ import annotations

import csv
import stat
from pathlib import Path

from report_cleanup.ml.features import FEATURE_NAMES, RAW_SIGNAL_NAMES
from training import (
    BASELINE_PREDICTION_COLUMN,
    BASELINE_THRESHOLD_COLUMN,
    LABEL_COLUMN,
)
from training.dataset import (
    PAIR_CSV_COLUMNS,
    SIGNAL_MISSING_COLUMNS,
    SIGNAL_PREFIX,
)
from training.generate_pairs import build_pair_rows, write_pair_csv


def _report(uid: int, name: str, fields: set[str], **overrides) -> dict:
    record = {
        "report_uid": uid,
        "report_name": name,
        "report_fields_set": fields,
        "business_objects_set": {"worker"},
        "built_in_prompts_set": set(),
        "related_bos_set": set(),
        "authorized_usage_set": set(),
        "data_source": "Workers",
        "report_type": "Advanced",
    }
    record.update(overrides)
    return record


def test_candidate_export_uses_blocking_and_never_fabricates_labels(cfg) -> None:
    records = [
        _report(1, "Active Worker", {"id", "name"}),
        _report(2, "Copy of Active Worker", {"id", "name"}),
        _report(
            3,
            "Unrelated Finance Listing",
            {"ledger", "amount"},
            business_objects_set={"finance"},
            data_source="Financials",
            report_type="Matrix",
        ),
    ]

    rows = build_pair_rows(records, cfg)

    assert len(rows) == 1
    row = rows[0]
    assert {row["report_uid_a"], row["report_uid_b"]} == {1, 2}
    assert row[LABEL_COLUMN] == ""
    assert BASELINE_PREDICTION_COLUMN in row
    assert BASELINE_PREDICTION_COLUMN in PAIR_CSV_COLUMNS
    assert row[BASELINE_PREDICTION_COLUMN] in (0, 1)
    assert row[BASELINE_THRESHOLD_COLUMN] == cfg.duplicate_thresholds["possible"]
    assert all(SIGNAL_PREFIX + name in row for name in RAW_SIGNAL_NAMES)
    assert all(name in row for name in SIGNAL_MISSING_COLUMNS)
    assert all(name in row for name in FEATURE_NAMES)


def test_export_preserves_real_zero_separately_from_unavailable_signal(cfg) -> None:
    records = [
        _report(1, "Worker Detail", {"id"}, data_source="Source A"),
        _report(2, "Worker Detail Copy", {"id"}, data_source="Source B"),
    ]

    [row] = build_pair_rows(records, cfg)

    assert row["sig_data_source"] == "0.0"
    assert row["sig_data_source_missing"] == 0
    assert row["sig_authorized_usage"] == ""
    assert row["sig_authorized_usage_missing"] == 1


def test_export_limit_is_deterministic(cfg) -> None:
    records = [
        _report(uid, f"Worker Detail {uid}", {"shared", str(uid)})
        for uid in range(5)
    ]

    first = build_pair_rows(records, cfg, limit=3)
    second = build_pair_rows(records, cfg, limit=3)

    assert first == second
    assert len(first) == 3
    assert all(row[LABEL_COLUMN] == "" for row in first)


def test_write_pair_csv_uses_canonical_order_blank_labels_and_private_permissions(
    cfg, tmp_path: Path
) -> None:
    rows = build_pair_rows(
        [
            _report(1, "Worker", {"id"}),
            _report(2, "Copy Worker", {"id"}),
        ],
        cfg,
    )

    path = write_pair_csv(rows, tmp_path / "private" / "pairs.csv")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
    assert tuple(reader.fieldnames or ()) == PAIR_CSV_COLUMNS
    assert len(written) == 1
    assert written[0][LABEL_COLUMN] == ""
    assert written[0][BASELINE_PREDICTION_COLUMN] in {"0", "1"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
