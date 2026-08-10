"""Model artifact persistence and schema validation.

What is saved is a state dict plus a plain-dict metadata block — never a pickled
`nn.Module`. Two reasons: unpickling an arbitrary object is a code-execution
vector for an artifact that may be copied between machines, and an opaque pickle
gives a reviewer no way to answer "what features did this thing train on?".

Loading is the enforcement point for feature-schema compatibility. If the
artifact's feature names, ordering, count, or schema version disagree with the
running `features.py`, the load fails loudly. Silently scoring a differently
ordered vector would produce confident, wrong duplicate probabilities — far worse
than no model at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .model import ARCHITECTURE, DuplicateMLP

ARTIFACT_FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when a model artifact is missing, unreadable, or malformed."""


class FeatureSchemaMismatchError(ArtifactError):
    """Raised when an artifact was trained on a different feature schema."""


@dataclass
class LoadedArtifact:
    """A ready-to-use model plus everything needed to audit where it came from."""

    model: DuplicateMLP
    metadata: dict[str, Any]
    path: Path

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.get("feature_names", []))

    @property
    def threshold(self) -> float:
        """Decision threshold chosen during training (probability, 0..1)."""
        return float(self.metadata.get("decision_threshold", 0.5))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))


def build_metadata(
    *,
    model: DuplicateMLP,
    model_version: str,
    decision_threshold: float,
    threshold_criterion: str,
    trained_at: str,
    seed: int,
    hyperparameters: dict[str, Any],
    dataset_counts: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the metadata block persisted alongside the weights.

    Everything here is a plain Python scalar / list / dict so the artifact can be
    read back under ``weights_only=True``.
    """
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "input_size": int(model.input_size),
        "hidden_sizes": [int(h) for h in model.hidden_sizes],
        "dropout": float(model.dropout),
        # Normalization is a fixed, code-defined transform (percent/100 with a
        # missingness indicator) — recorded so a reader does not have to guess.
        "normalization": "raw signals scaled percent/100 clamped to [0,1]; "
                         "missing signals encoded as value 0.0 with indicator 1.0",
        "decision_threshold": float(decision_threshold),
        "threshold_criterion": threshold_criterion,
        "trained_at": trained_at,
        # TorchVersion is a str subclass in recent releases but is not accepted by
        # the restricted weights-only unpickler. Persist a plain string.
        "torch_version": str(torch.__version__),
        "random_seed": int(seed),
        "hyperparameters": dict(hyperparameters),
        "dataset_counts": dict(dataset_counts),
        "evaluation_metrics": dict(evaluation_metrics),
        "notes": notes,
    }


def save_artifact(path: str | Path, model: DuplicateMLP, metadata: dict[str, Any]) -> Path:
    """Write ``{state_dict, metadata}`` to ``path``, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, p)
    return p


def _validate_schema(metadata: dict[str, Any]) -> None:
    saved_version = str(metadata.get("feature_schema_version", ""))
    if saved_version != FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaMismatchError(
            f"Model was trained on feature schema version {saved_version!r} but this "
            f"build uses {FEATURE_SCHEMA_VERSION!r}. Retrain the model."
        )

    saved_names = list(metadata.get("feature_names") or [])
    current = list(FEATURE_NAMES)
    if saved_names != current:
        raise FeatureSchemaMismatchError(
            "Model feature schema does not match this build.\n"
            f"  trained on ({len(saved_names)}): {saved_names}\n"
            f"  expected   ({len(current)}): {current}"
        )

    if metadata.get("input_size") != len(FEATURE_NAMES):
        raise FeatureSchemaMismatchError(
            f"Model input_size is {metadata.get('input_size')!r}; expected {len(FEATURE_NAMES)}."
        )

    saved_arch = str(metadata.get("architecture", ""))
    if saved_arch != ARCHITECTURE:
        raise FeatureSchemaMismatchError(
            f"Model architecture {saved_arch!r} is not the expected {ARCHITECTURE!r}."
        )


def load_artifact(path: str | Path) -> LoadedArtifact:
    """Load and validate an artifact. Raises `ArtifactError` on any problem.

    ``weights_only=True`` is used: the payload is tensors plus plain containers,
    so nothing needs arbitrary-object unpickling.
    """
    p = Path(path)
    if not p.exists():
        raise ArtifactError(f"Model artifact not found: {p}")
    if not p.is_file():
        raise ArtifactError(f"Model artifact path is not a file: {p}")

    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"Model artifact not found: {p}") from exc
    except Exception as exc:  # corrupt file, wrong format, disallowed globals
        raise ArtifactError(f"Could not read model artifact {p}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ArtifactError(f"Model artifact {p} has an unexpected layout (not a mapping).")

    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
        raise ArtifactError(
            f"Model artifact {p} is missing its 'state_dict' or 'metadata' block."
        )

    _validate_schema(metadata)

    try:
        model = DuplicateMLP(
            input_size=int(metadata["input_size"]),
            hidden_sizes=tuple(int(h) for h in metadata["hidden_sizes"]),
            dropout=float(metadata.get("dropout", 0.0)),
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactError(f"Model weights in {p} do not match the recorded architecture: {exc}") from exc

    model.eval()
    return LoadedArtifact(model=model, metadata=metadata, path=p)
