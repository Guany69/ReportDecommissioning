"""Startup config validation fails clearly on bad weights/thresholds/sections."""
import copy

import pytest

from report_cleanup.config import ConfigError, load_config, validate_config


def _data():
    # load_config already validates the bundled config; copy its parsed data.
    return copy.deepcopy(load_config()._data)


def test_bundled_config_is_valid():
    validate_config(_data())  # must not raise


def test_duplicate_weights_must_sum_to_one():
    d = _data()
    d["duplicate_weights"]["field_jaccard"] = 0.99  # breaks the sum
    with pytest.raises(ConfigError, match="duplicate_weights"):
        validate_config(d)


def test_missing_required_section_raises():
    d = _data()
    del d["overall"]
    with pytest.raises(ConfigError, match="Missing required"):
        validate_config(d)


def test_overlapping_recommendation_mins_raise():
    d = _data()
    d["recommendation"]["thresholds"].append({"min": 80, "label": "Dup"})
    with pytest.raises(ConfigError, match="overlapping"):
        validate_config(d)


def test_recommendation_must_cover_zero():
    d = _data()
    d["recommendation"]["thresholds"] = [{"min": 40, "label": "X"}, {"min": 80, "label": "Y"}]
    with pytest.raises(ConfigError, match="min:0"):
        validate_config(d)


def test_soft_cap_must_be_below_hard_score():
    d = _data()
    d["overall"]["soft_score_cap"] = 100
    with pytest.raises(ConfigError, match="soft_score_cap"):
        validate_config(d)


def test_bad_cadence_range_raises():
    d = _data()
    d["recurrence"]["cadences"]["weekly"] = {"min_days": 9, "max_days": 5}
    with pytest.raises(ConfigError, match="cadence"):
        validate_config(d)
