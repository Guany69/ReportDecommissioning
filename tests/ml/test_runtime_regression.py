"""Runtime ML-vs-baseline behavior and deterministic scoring regressions."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from report_cleanup.duplicate_similarity import (
    compute_duplicate_matches,
    compute_duplicate_similarity,
)
from report_cleanup.hard_rules import apply_hard_rules
from report_cleanup.ml.features import FEATURE_COUNT, FEATURE_NAMES
from report_cleanup.overall import calculate_overall_score


def _report(uid: int, name: str, fields: set[str], **overrides) -> dict:
    record = {
        "report_uid": uid,
        "report_name": name,
        "report_fields_set": fields,
        "business_objects_set": {"worker"},
        "built_in_prompts_set": {"as of date"},
        "related_bos_set": {"organization"},
        "authorized_usage_set": {"HR"},
        "data_source": "Workers",
        "report_type": "Advanced",
        "category": "HR",
        "report_tag": "Worker",
        "worklet": "No",
    }
    record.update(overrides)
    return record


@dataclass
class _FakeStats:
    pairs_scored: int = 0
    batches: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "pairs_scored": self.pairs_scored,
            "batches": self.batches,
            "inference_seconds": 0.0,
        }


class _FakePredictor:
    def __init__(self, probabilities: list[float], threshold: float = 0.8) -> None:
        self.probabilities = probabilities
        self.threshold = threshold
        self.model_version = "synthetic-runtime-v1"
        self.calls: list[list[list[float]]] = []
        self.stats = _FakeStats()

    def predict(self, vectors) -> list[float]:
        batch = [list(vector) for vector in vectors]
        self.calls.append(batch)
        assert len(batch) == len(self.probabilities)
        self.stats.pairs_scored += len(batch)
        self.stats.batches += int(bool(batch))
        return list(self.probabilities)

    def is_duplicate(self, probability: float) -> bool:
        return probability >= self.threshold


def test_disabled_ml_preserves_weighted_baseline_duplicate_verdict(cfg) -> None:
    records = [
        _report(1, "Active Worker", {"id", "name", "location"}),
        _report(2, "Copy of Active Worker", {"id", "name", "location"}),
    ]
    expected = compute_duplicate_similarity(records[0], records[1], cfg)

    stats = compute_duplicate_matches(records, cfg, predictor=None)

    assert expected.potential_duplicate is True
    assert records[0]["potential_duplicate"] is expected.potential_duplicate
    assert records[0]["duplicate_similarity"] == expected.overall
    assert records[0]["duplicate_relationship"] == expected.relationship
    assert records[0]["duplicate_scoring_mode"] == "weighted_baseline"
    assert records[0]["duplicate_model_status"] == "disabled"
    assert records[0]["duplicate_ml_probability"] is None
    assert records[0]["duplicate_ml_prediction"] is None
    assert stats["candidate_pair_count"] == 1
    assert stats["pairs_scored_by_ml"] == 0


def test_missing_model_fallback_status_is_explicit_while_using_same_baseline(cfg) -> None:
    records = [
        _report(1, "Worker Detail", {"id", "name"}),
        _report(2, "Worker Detail", {"id", "name"}),
    ]

    stats = compute_duplicate_matches(
        records, cfg, predictor=None, scoring_status="fallback_weighted"
    )

    assert all(record["potential_duplicate"] for record in records)
    assert all(record["duplicate_scoring_mode"] == "weighted_baseline" for record in records)
    assert all(record["duplicate_model_status"] == "fallback_weighted" for record in records)
    assert stats["model_status"] == "fallback_weighted"


def test_ml_probability_can_reject_pair_that_weighted_baseline_would_flag(cfg) -> None:
    original = [
        _report(1, "Active Worker", {"id", "name", "location"}),
        _report(2, "Copy of Active Worker", {"id", "name", "location"}),
    ]
    baseline = copy.deepcopy(original)
    compute_duplicate_matches(baseline, cfg)
    assert baseline[0]["potential_duplicate"] is True

    predictor = _FakePredictor([0.2], threshold=0.8)
    stats = compute_duplicate_matches(original, cfg, predictor=predictor)

    assert len(predictor.calls) == 1
    assert len(predictor.calls[0]) == 1
    assert len(predictor.calls[0][0]) == FEATURE_COUNT
    assert all(record["potential_duplicate"] is False for record in original)
    assert all(record["duplicate_ml_probability"] == pytest.approx(0.2) for record in original)
    assert all(record["duplicate_ml_threshold"] == pytest.approx(0.8) for record in original)
    assert all(record["duplicate_ml_prediction"] is False for record in original)
    assert all(record["duplicate_scoring_mode"] == "ml" for record in original)
    assert all(record["duplicate_model_status"] == "pytorch" for record in original)
    assert stats["pairs_scored_by_ml"] == 1


def test_ml_probability_can_accept_low_weighted_similarity_candidate(cfg) -> None:
    shared = {"shared"}
    records = [
        _report(
            1,
            "Apples Quarterly Dashboard",
            shared | {f"left-{i}" for i in range(9)},
            business_objects_set={"worker"},
            built_in_prompts_set=set(),
            related_bos_set=set(),
            authorized_usage_set=set(),
            data_source="Source A",
            report_type="Advanced",
        ),
        _report(
            2,
            "Oranges Annual Listing",
            shared | {f"right-{i}" for i in range(9)},
            business_objects_set={"finance"},
            built_in_prompts_set=set(),
            related_bos_set=set(),
            authorized_usage_set=set(),
            data_source="Source B",
            report_type="Matrix",
        ),
    ]
    baseline = copy.deepcopy(records)
    compute_duplicate_matches(baseline, cfg)
    assert baseline[0]["potential_duplicate"] is False

    predictor = _FakePredictor([0.91], threshold=0.85)
    compute_duplicate_matches(records, cfg, predictor=predictor)

    assert all(record["potential_duplicate"] is True for record in records)
    assert all(record["duplicate_relationship"] == "Possible Duplicate" for record in records)
    assert all(record["duplicate_ml_probability"] == pytest.approx(0.91) for record in records)
    assert all(record["duplicate_ml_prediction"] is True for record in records)
    assert any(
        "PyTorch duplicate model probability 91.0%" in reason.label
        for reason in records[0]["duplicate_reason_trail"]
    )


def test_runtime_batches_all_deterministic_candidate_pairs_in_one_predict_call(cfg) -> None:
    records = [
        _report(1, "Alpha Detail", {"shared", "a"}),
        _report(2, "Beta Detail", {"shared", "b"}),
        _report(3, "Gamma Detail", {"shared", "c"}),
    ]
    predictor = _FakePredictor([0.1, 0.9, 0.2], threshold=0.8)

    stats = compute_duplicate_matches(records, cfg, predictor=predictor)

    assert len(predictor.calls) == 1
    assert len(predictor.calls[0]) == 3
    assert stats["candidate_pair_count"] == 3
    assert stats["pairs_scored_by_ml"] == 3
    assert stats["batches"] == 1


def test_ml_feature_evidence_matches_canonical_normalized_model_input(cfg) -> None:
    records = [
        _report(1, "Worker One", {"id", "name"}, authorized_usage_set=set()),
        _report(2, "Worker Two", {"id", "manager"}, authorized_usage_set=set()),
    ]
    predictor = _FakePredictor([0.9])

    compute_duplicate_matches(records, cfg, predictor=predictor)
    evidence = records[0]["duplicate_feature_values"]

    assert tuple(evidence) == FEATURE_NAMES
    assert list(evidence.values()) == predictor.calls[0][0]
    assert all(0.0 <= value <= 1.0 for value in evidence.values())
    assert evidence["authorized_usage"] == 0.0
    assert evidence["authorized_usage_missing"] == 1.0


@pytest.mark.parametrize("use_ml", [False, True])
def test_duplicate_evidence_never_changes_numeric_overall_score(cfg, use_ml: bool) -> None:
    result = calculate_overall_score(
        cleanup_points=45,
        protection_credit=12,
        hard_rule_triggered=False,
        cfg=cfg,
    )
    records = [
        _report(1, "Worker Report", {"id", "name"}, overall_score=result.overall_score),
        _report(2, "Copy Worker Report", {"id", "name"}, overall_score=result.overall_score),
    ]
    predictor = _FakePredictor([0.99]) if use_ml else None

    compute_duplicate_matches(records, cfg, predictor=predictor)

    assert [record["overall_score"] for record in records] == [63, 63]


def test_hard_rules_remain_deterministic_and_independent_of_ml_configuration(cfg) -> None:
    dnu = apply_hard_rules({"report_name": "Finance DNU Archive"}, cfg)
    non_match = apply_hard_rules({"report_name": "Abundance Overview"}, cfg)
    orphan = apply_hard_rules(
        {
            "report_name": "Worker Card",
            "worklet": "Yes",
            "landing_page": None,
            "report_landing_page": None,
            "areas_used": None,
        },
        cfg,
    )

    assert dnu is not None and dnu.rule_id == "dnu_in_name"
    assert non_match is None
    assert orphan is not None and orphan.rule_id == "orphan_worklet"
