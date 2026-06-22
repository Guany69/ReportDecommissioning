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


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    validate_config(data)
    return Config(data, source=p)
