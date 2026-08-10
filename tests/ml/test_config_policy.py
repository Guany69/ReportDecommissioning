"""Configuration validation for the optional ML duplicate layer."""
from __future__ import annotations

import copy

import pytest

from report_cleanup.config import Config, ConfigError, load_config, validate_config


def _config_data() -> dict:
    return copy.deepcopy(load_config()._data)


def test_bundled_ml_configuration_is_disabled_and_has_explicit_fallback(cfg) -> None:
    assert cfg.ml_duplicate == {
        "enabled": False,
        "model_path": "models/duplicate_model.pt",
        "decision_threshold": None,
        "fallback_to_weighted_similarity": True,
        "batch_size": 512,
    }


def test_legacy_configuration_without_ml_section_remains_valid() -> None:
    data = _config_data()
    data.pop("ml_duplicate", None)

    validate_config(data)
    assert Config(data).ml_duplicate == {}


@pytest.mark.parametrize("threshold", [-0.01, 1.01, "0.8", True])
def test_invalid_ml_threshold_is_rejected(threshold) -> None:
    data = _config_data()
    data["ml_duplicate"]["decision_threshold"] = threshold

    with pytest.raises(ConfigError, match="decision_threshold"):
        validate_config(data)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, "32", True])
def test_invalid_ml_batch_size_is_rejected(batch_size) -> None:
    data = _config_data()
    data["ml_duplicate"]["batch_size"] = batch_size

    with pytest.raises(ConfigError, match="batch_size"):
        validate_config(data)


@pytest.mark.parametrize("key", ["enabled", "fallback_to_weighted_similarity"])
def test_ml_boolean_flags_require_actual_booleans(key: str) -> None:
    data = _config_data()
    data["ml_duplicate"][key] = 1

    with pytest.raises(ConfigError, match=key):
        validate_config(data)


@pytest.mark.parametrize("model_path", [None, "", "   ", 7])
def test_ml_model_path_must_be_nonempty_text(model_path) -> None:
    data = _config_data()
    data["ml_duplicate"]["model_path"] = model_path

    with pytest.raises(ConfigError, match="model_path"):
        validate_config(data)
