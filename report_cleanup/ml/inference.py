"""Production inference for the duplicate classifier.

`DuplicatePredictor` owns the loaded model for the lifetime of a pipeline run:
constructed once in `pipeline.run_pipeline`, handed to
`duplicate_similarity.compute_duplicate_matches`, and never re-read from disk.
Scoring is batched — one tensor per `batch_size` candidate pairs rather than one
forward pass per pair.

`build_predictor` is the only place the enable/fallback policy is decided, so
there is exactly one answer to "did ML actually run?" and it is recorded on every
record as ``duplicate_scoring_mode``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .artifact import ArtifactError, LoadedArtifact, load_artifact
from .features import FEATURE_COUNT, FEATURE_NAMES

# Values stamped onto every record so the output itself says which path ran.
SCORING_MODE_ML = "ml"
SCORING_MODE_BASELINE = "weighted_baseline"

DEFAULT_BATCH_SIZE = 512


class MLModelUnavailableError(RuntimeError):
    """ML duplicate scoring was required but no usable model could be loaded."""


@dataclass
class InferenceStats:
    """Cheap observability: enough to prove ML ran and how much it cost."""

    pairs_scored: int = 0
    batches: int = 0
    inference_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs_scored": self.pairs_scored,
            "batches": self.batches,
            "inference_seconds": round(self.inference_seconds, 4),
        }


@dataclass
class MLDuplicateSettings:
    """Resolved `ml_duplicate` configuration."""

    enabled: bool = False
    model_path: str = "models/duplicate_model.pt"
    decision_threshold: float | None = None      # None -> use the artifact's threshold
    fallback_to_weighted_similarity: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE

    @classmethod
    def from_config(cls, cfg) -> "MLDuplicateSettings":
        section = getattr(cfg, "ml_duplicate", None) or {}
        threshold = section.get("decision_threshold", None)
        return cls(
            enabled=bool(section.get("enabled", False)),
            model_path=str(section.get("model_path", cls.model_path)),
            decision_threshold=None if threshold is None else float(threshold),
            fallback_to_weighted_similarity=bool(
                section.get("fallback_to_weighted_similarity", True)),
            batch_size=int(section.get("batch_size", DEFAULT_BATCH_SIZE)),
        )


class DuplicatePredictor:
    """Batched duplicate-probability inference over pre-extracted feature vectors.

    The model is loaded once at construction. `predict` accepts the 18-dim vectors
    produced by `features.build_feature_vector` and returns probabilities in
    0.0..1.0 — the caller decides how to present them.
    """

    def __init__(
        self,
        artifact: LoadedArtifact,
        threshold: float | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        import torch  # local: report_cleanup must import without torch installed

        self._torch = torch
        self.artifact = artifact
        self.batch_size = max(1, int(batch_size))
        # CPU by default: the feature vectors are tiny and this keeps deployment
        # predictable. Callers may explicitly pass ``device="cuda"``.
        if device is None:
            device = "cpu"
        self.device = device

        self.model = artifact.model.to(self.device)
        self.model.eval()

        # A config threshold overrides the trained one (operators may want to be
        # stricter in production); otherwise the artifact's own value is used.
        self.threshold = float(artifact.threshold if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"decision_threshold must be in 0..1, got {self.threshold}")

        self.stats = InferenceStats()

    # -- metadata surface ---------------------------------------------------
    @property
    def model_version(self) -> str:
        return self.artifact.model_version

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.artifact.metadata)

    @property
    def feature_names(self) -> list[str]:
        return self.artifact.feature_names

    def describe(self) -> str:
        return (f"DuplicatePredictor(model_version={self.model_version!r}, "
                f"threshold={self.threshold:.4f}, device={self.device!r}, "
                f"path={self.artifact.path})")

    # -- inference ----------------------------------------------------------
    def predict(self, feature_vectors: Sequence[Sequence[float]]) -> list[float]:
        """Score N feature vectors -> N probabilities in 0..1, order preserved."""
        rows = list(feature_vectors)
        if not rows:
            return []
        for i, row in enumerate(rows):
            if len(row) != FEATURE_COUNT:
                raise ValueError(
                    f"Feature vector {i} has {len(row)} values, expected {FEATURE_COUNT} "
                    f"({', '.join(FEATURE_NAMES)})."
                )

        torch = self._torch
        out: list[float] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start:start + self.batch_size]
                x = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                probs = torch.sigmoid(self.model(x))
                out.extend(float(p) for p in probs.detach().cpu().tolist())
                self.stats.batches += 1
        self.stats.inference_seconds += time.perf_counter() - started
        self.stats.pairs_scored += len(rows)
        return out

    def predict_one(self, feature_vector: Sequence[float]) -> float:
        """Single-pair convenience wrapper. Prefer `predict` for bulk scoring."""
        return self.predict([feature_vector])[0]

    def is_duplicate(self, probability: float) -> bool:
        """Apply the decision threshold (inclusive at the boundary)."""
        return probability >= self.threshold


def resolve_model_path(raw_path: str, config_source: Path | None = None) -> Path:
    """Resolve a relative model path beside the active config (project root by default)."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    base = config_source.parent if config_source is not None else Path(__file__).resolve().parents[2]
    return base / p


def build_predictor(cfg, warn=None) -> DuplicatePredictor | None:
    """Apply the ml_duplicate policy and return a predictor, or None for baseline.

    Policy (documented in README and config.yaml):

    * disabled                                   -> None (deterministic baseline)
    * enabled, model loads                       -> DuplicatePredictor
    * enabled, model missing/invalid, fallback   -> warn loudly, return None
    * enabled, model missing/invalid, no fallback-> raise MLModelUnavailableError

    ``warn`` is an optional callback receiving a message string; the message is
    always also written to stderr so a headless run cannot hide the downgrade.
    """
    settings = MLDuplicateSettings.from_config(cfg)
    if not settings.enabled:
        return None

    path = resolve_model_path(settings.model_path, getattr(cfg, "source", None))
    try:
        artifact = load_artifact(path)
        return DuplicatePredictor(
            artifact,
            threshold=settings.decision_threshold,
            batch_size=settings.batch_size,
        )
    except (ArtifactError, ValueError, ImportError) as exc:
        message = (
            f"ML duplicate scoring is enabled but the model at {path} could not be used: {exc}"
        )
        if not settings.fallback_to_weighted_similarity:
            raise MLModelUnavailableError(
                message + " (ml_duplicate.fallback_to_weighted_similarity is false)"
            ) from exc
        message += " Falling back to the deterministic weighted similarity baseline."
        print(f"[warning] {message}", file=sys.stderr, flush=True)
        if warn is not None:
            warn(message)
        return None
