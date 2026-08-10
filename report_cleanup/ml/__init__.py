"""Machine-learning layer for duplicate detection.

Scope, deliberately narrow: the model answers one supervised question — *given a
candidate pair of reports, what is the probability they are duplicates /
consolidation candidates?* Everything else in the engine (hard rules, cleanup
risk, business-protection credits, the Overall Decommissioning Score, candidate
blocking, keeper selection) stays deterministic. The ML probability is duplicate
**evidence** and never moves the Overall Decommissioning Score.

`features` is import-safe without PyTorch, so the deterministic engine runs on a
machine that has no torch installed. `model`, `artifact`, and `inference` import
torch and are therefore imported lazily via the accessors below rather than at
package-import time.
"""
from __future__ import annotations

from .features import (FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_VERSION,
                       RAW_SIGNAL_NAMES, build_feature_vector,
                       extract_raw_signals, feature_vector_for_pair)

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "RAW_SIGNAL_NAMES",
    "build_feature_vector",
    "extract_raw_signals",
    "feature_vector_for_pair",
    "build_predictor",
    "DuplicatePredictor",
    "SCORING_MODE_BASELINE",
    "SCORING_MODE_ML",
]

# Torch-free constants that callers need even when ML is disabled.
SCORING_MODE_ML = "ml"
SCORING_MODE_BASELINE = "weighted_baseline"


def build_predictor(cfg, warn=None):
    """Lazy proxy for `inference.build_predictor` (imports torch on demand)."""
    # Preserve the deterministic deployment path even if training/inference
    # dependencies are intentionally absent from that environment.
    section = getattr(cfg, "ml_duplicate", None) or {}
    if not section.get("enabled", False):
        return None
    from .inference import build_predictor as _impl
    return _impl(cfg, warn=warn)


def __getattr__(name: str):
    """Expose the torch-backed classes without importing torch at package import."""
    if name == "DuplicatePredictor":
        from .inference import DuplicatePredictor
        return DuplicatePredictor
    if name == "MLModelUnavailableError":
        from .inference import MLModelUnavailableError
        return MLModelUnavailableError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
