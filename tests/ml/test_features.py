"""Canonical duplicate-pair feature extraction and schema tests."""
from __future__ import annotations

import pytest

from report_cleanup.ml.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    MISSING_SUFFIX,
    RAW_SIGNAL_NAMES,
    build_feature_vector,
    extract_raw_signals,
    feature_vector_for_pair,
    scalar_eq,
    set_containment,
    set_jaccard,
)


def _report(name: str | None, **overrides) -> dict:
    report = {
        "report_name": name,
        "report_fields_set": {"employee id", "name", "location"},
        "business_objects_set": {"worker", "organization"},
        "built_in_prompts_set": {"as of date", "company"},
        "related_bos_set": {"manager"},
        "data_source": "Workers",
        "authorized_usage_set": {"HR", "Payroll"},
        "report_type": "Advanced",
    }
    report.update(overrides)
    return report


def _feature_map(vector: list[float]) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, vector, strict=True))


def test_set_similarity_primitives_calculate_jaccard_and_containment() -> None:
    left = {"a", "b", "c"}
    right = {"b", "c", "d", "e"}

    assert set_jaccard(left, right) == pytest.approx(40.0)
    assert set_containment(left, right) == pytest.approx(200.0 / 3.0)


@pytest.mark.parametrize("left,right", [(set(), {"x"}), ({"x"}, set()), (set(), set())])
def test_set_similarity_is_unavailable_when_either_side_is_empty(left, right) -> None:
    assert set_jaccard(left, right) is None
    assert set_containment(left, right) is None


def test_scalar_equality_is_normalized_and_distinguishes_mismatch_from_missing() -> None:
    assert scalar_eq(" Advanced ", "advanced") == 100.0
    assert scalar_eq("Advanced", "Matrix") == 0.0
    assert scalar_eq("", "") is None
    assert scalar_eq(None, "Advanced") is None


def test_extract_raw_signals_includes_name_and_equality_features(cfg) -> None:
    left = _report("Copy of Active Worker Report")
    right = _report(
        "Active Worker",
        report_fields_set={"employee id", "name", "manager"},
        data_source=" workers ",
        report_type="Matrix",
    )

    raw = extract_raw_signals(left, right, cfg)

    assert tuple(raw) == RAW_SIGNAL_NAMES
    assert raw["field_jaccard"] == pytest.approx(50.0)
    assert raw["smaller_containment"] == pytest.approx(200.0 / 3.0)
    assert raw["name"] == pytest.approx(100.0)
    assert raw["data_source"] == 100.0
    assert raw["report_type"] == 0.0


def test_feature_vector_has_contractual_order_and_normalized_ranges(cfg) -> None:
    vector = feature_vector_for_pair(_report("Worker A"), _report("Worker B"), cfg)

    assert len(vector) == FEATURE_COUNT == len(FEATURE_NAMES) == 2 * len(RAW_SIGNAL_NAMES)
    assert FEATURE_NAMES == tuple(
        item
        for raw_name in RAW_SIGNAL_NAMES
        for item in (raw_name, raw_name + MISSING_SUFFIX)
    )
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_feature_order_does_not_depend_on_raw_mapping_insertion_order() -> None:
    ascending = {name: float(index) for index, name in enumerate(RAW_SIGNAL_NAMES)}
    descending = dict(reversed(tuple(ascending.items())))

    assert build_feature_vector(ascending) == build_feature_vector(descending)


def test_missing_indicator_separates_unavailable_from_real_zero() -> None:
    raw = {name: 50.0 for name in RAW_SIGNAL_NAMES}
    raw["field_jaccard"] = None
    raw["data_source"] = 0.0

    features = _feature_map(build_feature_vector(raw))

    assert features["field_jaccard"] == 0.0
    assert features["field_jaccard_missing"] == 1.0
    assert features["data_source"] == 0.0
    assert features["data_source_missing"] == 0.0


def test_build_feature_vector_clamps_percentages_and_marks_omitted_values_missing() -> None:
    raw = {"field_jaccard": 125.0, "smaller_containment": -5.0}
    features = _feature_map(build_feature_vector(raw))

    assert features["field_jaccard"] == 1.0
    assert features["field_jaccard_missing"] == 0.0
    assert features["smaller_containment"] == 0.0
    assert features["smaller_containment_missing"] == 0.0
    assert features["business_object"] == 0.0
    assert features["business_object_missing"] == 1.0


def test_pair_feature_extraction_is_symmetric(cfg) -> None:
    left = _report(
        "Worker Detail",
        report_fields_set={"employee id", "name"},
        authorized_usage_set=set(),
    )
    right = _report(
        "Worker Details",
        report_fields_set={"employee id", "location", "manager"},
        authorized_usage_set={"HR"},
    )

    assert feature_vector_for_pair(left, right, cfg) == feature_vector_for_pair(right, left, cfg)
