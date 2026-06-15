import numpy as np
import pandas as pd

from report_cleanup.clean import (is_false, is_null, is_true, normalize_name,
                                   to_bool, to_date, to_number)


def test_is_null_variants():
    assert is_null(None) and is_null("") and is_null("   ") and is_null(np.nan) and is_null(pd.NaT)
    assert not is_null("x") and not is_null(0)


def test_bool_helpers():
    assert is_true("Yes") and is_true("TRUE") and is_true("1")
    assert is_false("No") and is_false("false") and is_false("0")
    assert not is_true("maybe")


def test_to_number():
    assert to_number("1,234") == 1234
    assert to_number("  42 ") == 42
    assert np.isnan(to_number("abc")) and np.isnan(to_number(""))


def test_to_date_iso_and_serial():
    assert to_date("2020-01-15").year == 2020
    # Excel serial 43831 == 2020-01-01
    assert to_date("43831").year == 2020
    assert pd.isna(to_date(""))


def test_to_bool(cfg):
    assert to_bool("yes", cfg.clean) is True
    assert to_bool("no", cfg.clean) is False
    assert pd.isna(to_bool("???", cfg.clean))


def test_normalize_name_strips_noise(cfg):
    noise = cfg.clean["name_noise"]
    assert normalize_name("2019 Annual Performance (Copy)", noise) == "annual performance"
    assert normalize_name("Annual Performance v2", noise) == "annual performance"
