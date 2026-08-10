"""The ML layer must not disturb the deterministic engine.

Two guarantees are under test here, and they are the ones that matter most for a
system whose recommendations retire real reports:

1. With ML disabled, duplicate detection behaves exactly as it did before the
   classifier existed — same flags, same relationships, same reason trails.
2. With ML enabled and scoring pairs, the Overall Decommissioning Score, hard
   rules, cleanup risk, and business-protection credits are byte-identical.
   Duplicate evidence is evidence; it never moves the number.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from report_cleanup.duplicate_similarity import (compute_duplicate_matches,
                                                 compute_duplicate_similarity)
from report_cleanup.ml.artifact import build_metadata, save_artifact
from report_cleanup.ml.features import FEATURE_COUNT
from report_cleanup.ml.inference import DuplicatePredictor
from report_cleanup.ml.model import build_model
from report_cleanup.overall import calculate_overall_score
from report_cleanup.hard_rules import apply_hard_rules
from report_cleanup.soft_scoring import score_report


def _report(uid: int, name: str, fields: set[str], **overrides) -> dict:
    record = {
        "report_uid": uid,
        "report_name": name,
        "report_fields_set": set(fields),
        "business_objects_set": {"worker"},
        "built_in_prompts_set": set(),
        "related_bos_set": set(),
        "authorized_usage_set": set(),
        "data_source": "Workers",
        "report_type": "Advanced",
        "field_extraction_status": "Matched",
    }
    record.update(overrides)
    return record


def _records() -> list[dict]:
    return [
        _report(1, "Active Worker Roster", {"id", "name", "dept"}),
        _report(2, "Active Worker Roster", {"id", "name", "dept"}),
        _report(3, "Quarterly Ledger Summary", {"ledger", "amount"},
                business_objects_set={"finance"}, data_source="Financials",
                report_type="Matrix"),
    ]


# ---- 1. ML disabled leaves the deterministic path untouched -----------------
def test_disabled_ml_reproduces_the_weighted_baseline_exactly(cfg) -> None:
    """compute_duplicate_matches(predictor=None) is the pre-ML behaviour."""
    records = _records()
    stats = compute_duplicate_matches(records, cfg, predictor=None)

    assert stats["duplicate_scoring_mode"] == "weighted_baseline"
    assert stats["pairs_scored_by_ml"] == 0

    for r in records:
        assert r["duplicate_scoring_mode"] == "weighted_baseline"
        assert r["duplicate_model_status"] == "disabled"
        assert r["duplicate_ml_probability"] is None
        assert r["duplicate_ml_prediction"] is None
        assert r["duplicate_model_version"] is None

    # The verdicts still come from the weighted formula.
    a, b, c = records
    assert a["potential_duplicate"] is True
    assert b["potential_duplicate"] is True
    assert c["potential_duplicate"] is False
    assert a["duplicate_relationship"] == "Nearly Identical"

    # And they agree with a direct call to the pairwise function.
    direct = compute_duplicate_similarity(a, b, cfg)
    assert a["duplicate_similarity"] == direct.overall
    assert a["duplicate_relationship"] == direct.relationship


def test_disabled_ml_keeps_the_weighted_component_reason_trail(cfg) -> None:
    records = _records()
    compute_duplicate_matches(records, cfg, predictor=None)

    trail = [reason.label for reason in records[0]["duplicate_reason_trail"]]
    assert any(label.startswith("Weighted duplicate similarity") for label in trail)
    assert any("Field Jaccard similarity" in label for label in trail)
    # No ML claim may appear when no model ran.
    assert not any("PyTorch" in label for label in trail)


# ---- 2. ML enabled does not move the Overall Decommissioning Score ----------
def _scoring_snapshot(record: dict, cfg) -> dict:
    """The deterministic half of the engine, computed from one record."""
    hit = apply_hard_rules(record, cfg)
    risk = score_report(record, cfg)
    overall = calculate_overall_score(risk.total_risk_score, 0, bool(hit), cfg)
    return {
        "hard_rule": bool(hit),
        "cleanup_risk_points": risk.total_risk_score,
        "usage_risk": risk.usage_risk,
        "age_risk": risk.age_risk,
        "usage_context_risk": risk.usage_context_risk,
        "overall_score": overall.overall_score,
        "cleanup_percentage": overall.cleanup_percentage,
    }


def test_ml_scoring_does_not_change_overall_score_or_hard_rules(
    cfg, always_positive_predictor: DuplicatePredictor
) -> None:
    """A model that flags every pair must still leave the score untouched."""
    records = _records()
    before = [_scoring_snapshot(copy.deepcopy(r), cfg) for r in records]

    stats = compute_duplicate_matches(records, cfg, predictor=always_positive_predictor,
                                      scoring_status="pytorch")
    assert stats["duplicate_scoring_mode"] == "ml"
    assert stats["pairs_scored_by_ml"] == stats["candidate_pair_count"] > 0

    after = [_scoring_snapshot(r, cfg) for r in records]
    assert after == before, "duplicate evidence must not alter deterministic scoring"

    # None of the ML fields is a scoring field.
    for r in records:
        assert "overall_score" not in r
        assert r["duplicate_scoring_mode"] == "ml"


def test_ml_flags_pairs_the_baseline_rejected_without_touching_scores(
    cfg, always_positive_predictor: DuplicatePredictor
) -> None:
    """The model owns the flag decision; the rules still own the label.

    The pair used here shares one field out of nineteen, so blocking makes it a
    candidate but the weighted formula's ceiling short-circuit refuses to flag it.
    That is exactly the case where a learned layer can differ from the heuristic.
    """
    records = [
        _report(1, "Apples Quarterly Dashboard",
                {f"a{i}" for i in range(9)} | {"shared"}),
        _report(2, "Oranges Annual Listing",
                {f"b{i}" for i in range(9)} | {"shared"}),
    ]
    baseline_records = copy.deepcopy(records)
    baseline_stats = compute_duplicate_matches(baseline_records, cfg, predictor=None)
    assert baseline_stats["candidate_pair_count"] == 1      # blocking DID pair them
    assert baseline_records[0]["potential_duplicate"] is False
    assert baseline_records[0]["duplicate_relationship"] == "Not Flagged"

    compute_duplicate_matches(records, cfg, predictor=always_positive_predictor)

    unrelated = records[0]
    assert unrelated["potential_duplicate"] is True          # model overrode the heuristic
    assert unrelated["duplicate_ml_prediction"] is True
    assert 0.0 <= unrelated["duplicate_ml_probability"] <= 1.0
    # ...but the descriptive relationship still comes from the deterministic rules,
    # falling back to an EXISTING label rather than a new ML-only one.
    assert unrelated["duplicate_relationship"] in {
        "Nearly Identical", "Smaller Report Contained in Larger Report",
        "High Field Overlap", "Possible Duplicate", "Likely Duplicate (Name Match)",
    }


def test_ml_reason_trail_leads_with_probability_and_keeps_feature_evidence(
    cfg, always_positive_predictor: DuplicatePredictor
) -> None:
    records = _records()
    compute_duplicate_matches(records, cfg, predictor=always_positive_predictor)

    trail = [reason.label for reason in records[0]["duplicate_reason_trail"]]
    assert trail[0].startswith("PyTorch duplicate model probability")
    assert "decision threshold" in trail[0]
    # The weighted evidence is retained underneath as reviewer context.
    assert any("Field Jaccard similarity" in label for label in trail)

    features = records[0]["duplicate_feature_values"]
    assert len(features) == FEATURE_COUNT
    assert features["field_jaccard_missing"] == 0.0


def test_model_is_loaded_once_not_once_per_candidate_pair(
    cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring N pairs must not mean N disk reads or N model constructions."""
    from report_cleanup.ml import artifact as artifact_module
    from report_cleanup.ml import inference as inference_module

    path = tmp_path / "duplicate_model.pt"
    model = build_model()
    save_artifact(path, model, build_metadata(
        model=model, model_version="load-once-test", decision_threshold=0.5,
        threshold_criterion="fixed for test", trained_at="2026-01-01T00:00:00+00:00",
        seed=0, hyperparameters={}, dataset_counts={}, evaluation_metrics={}))

    loads = {"count": 0}
    real_load = artifact_module.load_artifact

    def counting_load(p):
        loads["count"] += 1
        return real_load(p)

    monkeypatch.setattr(inference_module, "load_artifact", counting_load)

    class _Cfg:
        ml_duplicate = {"enabled": True, "model_path": str(path),
                        "fallback_to_weighted_similarity": False}
        source = None

    predictor = inference_module.build_predictor(_Cfg())
    assert loads["count"] == 1

    records = _records()
    stats = compute_duplicate_matches(records, cfg, predictor=predictor)
    assert stats["pairs_scored_by_ml"] >= 1
    assert loads["count"] == 1, "the artifact must not be re-read while scoring pairs"
    # And the whole candidate set went through a small number of batched calls.
    assert predictor.stats.batches <= 1 + stats["pairs_scored_by_ml"] // predictor.batch_size
