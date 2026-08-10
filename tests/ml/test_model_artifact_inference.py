"""PyTorch model, artifact, and production predictor tests."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from report_cleanup.ml.artifact import (
    ArtifactError,
    FeatureSchemaMismatchError,
    LoadedArtifact,
    build_metadata,
    load_artifact,
    save_artifact,
)
from report_cleanup.ml.features import FEATURE_COUNT, FEATURE_NAMES
from report_cleanup.ml.inference import (
    DuplicatePredictor,
    MLModelUnavailableError,
    build_predictor,
)
from report_cleanup.ml.model import DuplicateMLP, probabilities_from_logits


def _metadata(model: DuplicateMLP, *, threshold: float = 0.8) -> dict:
    return build_metadata(
        model=model,
        model_version="synthetic-test-v1",
        decision_threshold=threshold,
        threshold_criterion="synthetic test threshold only",
        trained_at="2026-08-10T12:00:00+00:00",
        seed=17,
        hyperparameters={"epochs": 2, "learning_rate": 0.001},
        dataset_counts={"train": 8, "validation": 2, "test": 2},
        evaluation_metrics={"validation": {"precision": 1.0}},
        notes="Synthetic fixture; never production evidence.",
    )


def _loaded_artifact(tmp_path: Path, *, threshold: float = 0.8) -> LoadedArtifact:
    model = DuplicateMLP()
    path = tmp_path / "synthetic.pt"
    save_artifact(path, model, _metadata(model, threshold=threshold))
    return load_artifact(path)


def _constant_probability_model(probability: float) -> DuplicateMLP:
    model = DuplicateMLP()
    logit = math.log(probability / (1.0 - probability))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        final = model.net[-1]
        assert isinstance(final, nn.Linear)
        final.bias.fill_(logit)
    return model


def _settings(**overrides) -> SimpleNamespace:
    values = {
        "enabled": True,
        "model_path": "unused.pt",
        "decision_threshold": None,
        "fallback_to_weighted_similarity": True,
        "batch_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(ml_duplicate=values)


def test_model_layers_and_batch_output_shape() -> None:
    model = DuplicateMLP(input_size=FEATURE_COUNT, hidden_sizes=(32, 16))
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (FEATURE_COUNT, 32),
        (32, 16),
        (16, 1),
    ]
    assert model(torch.zeros(7, FEATURE_COUNT)).shape == (7,)
    assert model(torch.zeros(FEATURE_COUNT)).shape == ()


def test_forward_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(5)
    model = DuplicateMLP()
    inputs = torch.rand(6, FEATURE_COUNT, dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float32)

    logits = model(inputs)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_probability_conversion_is_bounded_and_monotonic() -> None:
    probabilities = probabilities_from_logits(torch.tensor([-100.0, 0.0, 100.0]))

    assert probabilities.tolist()[1] == pytest.approx(0.5)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] < probabilities[1] < probabilities[2]


def test_artifact_save_load_round_trip_preserves_weights_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = DuplicateMLP(hidden_sizes=(8, 4), dropout=0.1)
    metadata = _metadata(model, threshold=0.87)

    path = save_artifact(tmp_path / "nested" / "duplicate.pt", model, metadata)
    loaded = load_artifact(path)

    assert path.exists()
    assert loaded.path == path
    assert loaded.threshold == pytest.approx(0.87)
    assert loaded.model_version == "synthetic-test-v1"
    assert loaded.feature_names == list(FEATURE_NAMES)
    assert loaded.model.training is False
    assert loaded.metadata["normalization"]
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)


def test_artifact_loader_explicitly_uses_safe_weights_only_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = DuplicateMLP()
    path = save_artifact(tmp_path / "safe.pt", model, _metadata(model))
    real_load = torch.load
    calls: list[dict] = []

    def recording_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr("report_cleanup.ml.artifact.torch.load", recording_load)

    load_artifact(path)

    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    "metadata_change",
    [
        {"feature_schema_version": "obsolete"},
        {"feature_names": list(reversed(FEATURE_NAMES))},
        {"architecture": "different_network"},
    ],
)
def test_artifact_rejects_incompatible_feature_or_model_schema(
    tmp_path: Path, metadata_change: dict
) -> None:
    model = DuplicateMLP()
    metadata = _metadata(model)
    metadata.update(metadata_change)
    path = save_artifact(tmp_path / "mismatch.pt", model, metadata)

    with pytest.raises(FeatureSchemaMismatchError):
        load_artifact(path)


def test_artifact_missing_and_corrupt_files_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path / "missing.pt")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a pytorch artifact")
    with pytest.raises(ArtifactError, match="Could not read"):
        load_artifact(corrupt)


def test_predictor_supports_single_and_batched_inference(tmp_path: Path) -> None:
    model = _constant_probability_model(0.75)
    path = save_artifact(tmp_path / "constant.pt", model, _metadata(model, threshold=0.7))
    predictor = DuplicatePredictor(load_artifact(path), batch_size=2, device="cpu")
    vector = [0.0] * FEATURE_COUNT

    single = predictor.predict_one(vector)
    batch = predictor.predict([vector, [1.0] * FEATURE_COUNT, vector])

    assert single == pytest.approx(0.75, abs=1e-6)
    assert batch == pytest.approx([0.75, 0.75, 0.75], abs=1e-6)
    assert all(0.0 <= probability <= 1.0 for probability in batch)
    assert predictor.stats.pairs_scored == 4
    assert predictor.stats.batches == 3  # one single batch, then two 2-row batches
    assert predictor.model.training is False


def test_predictor_threshold_is_inclusive_and_config_can_override_artifact(tmp_path: Path) -> None:
    artifact = _loaded_artifact(tmp_path, threshold=0.85)

    trained_threshold = DuplicatePredictor(artifact, device="cpu")
    overridden = DuplicatePredictor(artifact, threshold=0.9, device="cpu")

    assert trained_threshold.threshold == pytest.approx(0.85)
    assert trained_threshold.is_duplicate(0.85) is True
    assert trained_threshold.is_duplicate(0.84999) is False
    assert overridden.threshold == pytest.approx(0.9)


def test_predictor_rejects_wrong_feature_width_and_invalid_threshold(tmp_path: Path) -> None:
    artifact = _loaded_artifact(tmp_path)
    predictor = DuplicatePredictor(artifact, device="cpu")

    with pytest.raises(ValueError, match=f"expected {FEATURE_COUNT}"):
        predictor.predict([[0.0] * (FEATURE_COUNT - 1)])
    with pytest.raises(ValueError, match="decision_threshold"):
        DuplicatePredictor(artifact, threshold=1.1, device="cpu")


def test_build_predictor_loads_artifact_only_once_for_repeated_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = _loaded_artifact(tmp_path)
    calls: list[Path] = []

    def fake_load(path: Path) -> LoadedArtifact:
        calls.append(path)
        return artifact

    monkeypatch.setattr("report_cleanup.ml.inference.load_artifact", fake_load)
    predictor = build_predictor(_settings(model_path=str(artifact.path)))
    assert predictor is not None

    predictor.predict([[0.0] * FEATURE_COUNT])
    predictor.predict([[1.0] * FEATURE_COUNT])

    assert calls == [artifact.path]


def test_build_predictor_disabled_does_not_try_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load(path: Path) -> LoadedArtifact:
        raise AssertionError(f"disabled inference tried to load {path}")

    monkeypatch.setattr("report_cleanup.ml.inference.load_artifact", unexpected_load)
    assert build_predictor(_settings(enabled=False)) is None


def test_missing_model_warns_and_falls_back_when_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    warnings: list[str] = []
    predictor = build_predictor(
        _settings(model_path=str(tmp_path / "missing.pt")), warn=warnings.append
    )

    assert predictor is None
    assert len(warnings) == 1
    assert "Falling back" in warnings[0]
    assert "Falling back" in capsys.readouterr().err


def test_missing_model_fails_when_fallback_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(MLModelUnavailableError, match="fallback_to_weighted_similarity is false"):
        build_predictor(
            _settings(
                model_path=str(tmp_path / "missing.pt"),
                fallback_to_weighted_similarity=False,
            )
        )
