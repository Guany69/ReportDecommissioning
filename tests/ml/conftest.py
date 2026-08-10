"""Synthetic ML fixtures.

Every model built here has hand-set weights, not trained ones. They exist to
exercise plumbing — batching, thresholds, fallback, schema validation — and their
outputs carry no information about real duplicate detection.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from report_cleanup.ml.artifact import build_metadata, save_artifact
from report_cleanup.ml.inference import DuplicatePredictor
from report_cleanup.ml.model import build_model


def _constant_output_model(logit: float):
    """A model whose output is a fixed logit regardless of input.

    Achieved by zeroing every weight and putting the desired value in the final
    bias — deterministic, and it makes threshold/batching assertions independent of
    whatever the randomly initialized weights would have produced.
    """
    model = build_model()
    with torch.no_grad():
        for module in model.net:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        final = [m for m in model.net if isinstance(m, torch.nn.Linear)][-1]
        final.bias.fill_(logit)
    model.eval()
    return model


def write_constant_artifact(path: Path, logit: float, threshold: float,
                            model_version: str = "synthetic-smoke-test") -> Path:
    """Persist a constant-output model as a real artifact on disk."""
    model = _constant_output_model(logit)
    return save_artifact(path, model, build_metadata(
        model=model,
        model_version=model_version,
        decision_threshold=threshold,
        threshold_criterion="fixed by a test fixture; not a trained threshold",
        trained_at="2026-01-01T00:00:00+00:00",
        seed=0,
        hyperparameters={"synthetic": True},
        dataset_counts={"synthetic": True},
        evaluation_metrics={},
        notes="Synthetic fixture. Carries no information about real duplicates.",
    ))


@pytest.fixture
def always_positive_predictor(tmp_path: Path) -> DuplicatePredictor:
    """A predictor that returns ~1.0 for every pair (logit +8 -> p ~ 0.99966)."""
    from report_cleanup.ml.artifact import load_artifact

    path = write_constant_artifact(tmp_path / "always_positive.pt", logit=8.0, threshold=0.5)
    return DuplicatePredictor(load_artifact(path))


@pytest.fixture
def always_negative_predictor(tmp_path: Path) -> DuplicatePredictor:
    """A predictor that returns ~0.0 for every pair (logit -8)."""
    from report_cleanup.ml.artifact import load_artifact

    path = write_constant_artifact(tmp_path / "always_negative.pt", logit=-8.0, threshold=0.5)
    return DuplicatePredictor(load_artifact(path))
