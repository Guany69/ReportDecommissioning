"""Load and access config.yaml. The engine reads tunables only from here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Config:
    """Thin typed-ish wrapper over the parsed YAML with dotted access."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    # -- dotted getters -----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    # -- convenience accessors used across the engine ----------------------
    @property
    def aliases(self) -> dict[str, list[str]]:
        return self._data["aliases"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self._data["scoring"]

    @property
    def bands(self) -> list[dict[str, Any]]:
        return self._data["bands"]

    @property
    def dedup(self) -> dict[str, Any]:
        return self._data["dedup"]

    @property
    def flags(self) -> dict[str, Any]:
        return self._data["flags"]

    @property
    def clean(self) -> dict[str, Any]:
        return self._data["clean"]

    # -- new scoring-overhaul sections -------------------------------------
    @property
    def overall(self) -> dict[str, Any]:
        return self._data["overall"]

    @property
    def business_protection(self) -> dict[str, Any]:
        return self._data["business_protection"]

    @property
    def recurrence(self) -> dict[str, Any]:
        return self._data["recurrence"]

    @property
    def recommendation(self) -> dict[str, Any]:
        return self._data["recommendation"]

    @property
    def duplicate_weights(self) -> dict[str, Any]:
        return self._data["duplicate_weights"]

    @property
    def duplicate_thresholds(self) -> dict[str, Any]:
        return self._data["duplicate_thresholds"]

    @property
    def hard_rules(self) -> dict[str, Any]:
        return self._data.get("hard_rules", {})

    @property
    def ml_duplicate(self) -> dict[str, Any]:
        """PyTorch duplicate-classifier settings.

        Optional so an older config.yaml still loads; absent means disabled, which
        keeps the engine on the deterministic weighted baseline.
        """
        return self._data.get("ml_duplicate", {}) or {}

    def snapshot_json(self) -> str:
        """Serialized config stored alongside each run for auditability."""
        return json.dumps(self._data, sort_keys=True, default=str)


class ConfigError(ValueError):
    """Raised when config.yaml fails startup validation."""


_REQUIRED_SECTIONS = (
    "aliases", "clean", "scoring", "dedup", "flags",
    "overall", "business_protection", "recurrence", "recommendation",
    "duplicate_weights", "duplicate_thresholds",
)


def validate_config(data: dict[str, Any]) -> None:
    """Fail clearly on startup for misconfigured weights, thresholds, or caps.

    Validates: required sections present; duplicate weights total 1.0 (or 100);
    recommendation thresholds cover 0 with no gaps/overlaps; score caps sane.
    """
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping.")

    missing = [s for s in _REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigError(f"Missing required config section(s): {', '.join(missing)}.")

    # -- duplicate weights must total 1.0 (accept a 0..100 scale too) ----------
    weights = data["duplicate_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ConfigError("duplicate_weights must be a non-empty mapping.")
    total = float(sum(weights.values()))
    if not (abs(total - 1.0) < 1e-6 or abs(total - 100.0) < 1e-6):
        raise ConfigError(
            f"duplicate_weights must sum to 1.0 or 100 (got {total})."
        )

    # -- overall score caps --------------------------------------------------
    o = data["overall"]
    cap = o.get("soft_score_cap")
    hard = o.get("hard_rule_score")
    cmax = o.get("cleanup_risk_max")
    if not (isinstance(cmax, (int, float)) and cmax > 0):
        raise ConfigError("overall.cleanup_risk_max must be a positive number.")
    if not (isinstance(cap, (int, float)) and isinstance(hard, (int, float))):
        raise ConfigError("overall.soft_score_cap and overall.hard_rule_score must be numbers.")
    if cap >= hard:
        raise ConfigError(
            f"overall.soft_score_cap ({cap}) must be below hard_rule_score ({hard}) "
            "so 100 stays reserved for hard rules."
        )

    # -- business protection cap ---------------------------------------------
    bp = data["business_protection"]
    if not isinstance(bp.get("max_credit"), (int, float)) or bp["max_credit"] < 0:
        raise ConfigError("business_protection.max_credit must be a non-negative number.")

    # -- recommendation thresholds: descending, cover 0, no duplicate mins ----
    thresholds = (data["recommendation"] or {}).get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ConfigError("recommendation.thresholds must be a non-empty list.")
    mins = [t.get("min") for t in thresholds]
    if any(not isinstance(m, (int, float)) for m in mins):
        raise ConfigError("Every recommendation threshold needs a numeric 'min'.")
    if len(set(mins)) != len(mins):
        raise ConfigError(f"recommendation.thresholds has overlapping 'min' values: {mins}.")
    if min(mins) != 0:
        raise ConfigError("recommendation.thresholds must include a min:0 band so every score maps.")

    # -- recurrence cadence ranges -------------------------------------------
    cadences = (data["recurrence"] or {}).get("cadences") or {}
    for name, rng in cadences.items():
        lo, hi = rng.get("min_days"), rng.get("max_days")
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo <= hi):
            raise ConfigError(f"recurrence cadence '{name}' needs min_days <= max_days.")

    # -- ML duplicate classifier (optional section) ---------------------------
    # Not in _REQUIRED_SECTIONS: a config predating the ML layer must still load,
    # and its absence simply means the deterministic baseline is used.
    if "ml_duplicate" in data:
        _validate_ml_duplicate(data["ml_duplicate"])


def _validate_ml_duplicate(ml: Any) -> None:
    """Fail clearly on a malformed ml_duplicate section.

    Catches the settings that would otherwise surface as a confusing runtime error
    deep inside inference — a non-numeric threshold, a threshold outside 0..1, or a
    non-positive batch size.
    """
    if not isinstance(ml, dict):
        raise ConfigError("ml_duplicate must be a mapping.")

    if not isinstance(ml.get("enabled", False), bool):
        raise ConfigError("ml_duplicate.enabled must be true or false.")
    if not isinstance(ml.get("fallback_to_weighted_similarity", True), bool):
        raise ConfigError("ml_duplicate.fallback_to_weighted_similarity must be true or false.")

    model_path = ml.get("model_path", "")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ConfigError("ml_duplicate.model_path must be a non-empty string.")

    # null means "use the threshold stored in the trained model artifact".
    threshold = ml.get("decision_threshold", None)
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ConfigError("ml_duplicate.decision_threshold must be a number in 0..1, or null.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ConfigError(
                f"ml_duplicate.decision_threshold must be a probability in 0..1 (got {threshold}). "
                "It is a probability, not a percentage."
            )

    batch_size = ml.get("batch_size", 512)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigError("ml_duplicate.batch_size must be a positive integer.")


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    validate_config(data)
    return Config(data, source=p)
