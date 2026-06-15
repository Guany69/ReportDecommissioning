"""Value cleaning + the shared null/bool helpers every rule relies on.

A single definition of "null", "true", "false", "date", "number", and
"normalized name" keeps the scoring rules consistent across the engine.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---- scalar helpers -------------------------------------------------------
def is_null(v) -> bool:
    """True for NaN/None, empty, or whitespace-only values."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip() == ""


def text(v) -> str:
    return "" if is_null(v) else str(v).strip()


def is_true(v) -> bool:
    return text(v).lower() in {"yes", "y", "true", "1", "t"}


def is_false(v) -> bool:
    return text(v).lower() in {"no", "n", "false", "0", "f"}


def to_bool(v, cfg_clean):
    s = text(v).lower()
    if s in set(cfg_clean["truthy"]):
        return True
    if s in set(cfg_clean["falsy"]):
        return False
    return np.nan


def to_number(v):
    if is_null(v):
        return np.nan
    s = re.sub(r"[,\s$]", "", str(v))
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_date(v):
    """Parse to pandas Timestamp; supports ISO/US strings and Excel serials."""
    if is_null(v):
        return pd.NaT
    s = str(v).strip()
    # Excel serial number (days since 1899-12-30).
    if re.fullmatch(r"\d{4,6}(\.\d+)?", s):
        try:
            n = float(s)
            if 10000 < n < 80000:
                return pd.Timestamp("1899-12-30") + pd.to_timedelta(n, unit="D")
        except ValueError:
            pass
    ts = pd.to_datetime(s, errors="coerce")
    return ts


def normalize_text(v, null_tokens) -> object:
    """Trim, collapse internal whitespace; map configured null tokens to NaN.

    Short-circuits on is_null so pandas NA/NaT/NaN never leak through as the
    literal string "nan" (which would defeat every downstream 'is null' check).
    """
    if is_null(v):
        return np.nan
    s = re.sub(r"\s+", " ", str(v).strip())
    if s.lower() in {t.lower() for t in null_tokens}:
        return np.nan
    return s if s != "" else np.nan


def normalize_report_name(value: object) -> str:
    """Exact report-name key for joining Comprehensive <-> Runs and Where_Used.

    Case-insensitive, whitespace-trimmed, internal whitespace collapsed — and
    NOTHING else. Unlike ``normalize_name`` this is *not* de-noised: it never
    strips version/junk tokens, so two genuinely different reports can never
    collapse onto the same key. The two files are guaranteed to carry identical
    report names, so an exact (non-fuzzy) match is correct here.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().casefold().split())


def normalize_multiline(v, null_tokens) -> object:
    """Like normalize_text but PRESERVES line breaks.

    Used for columns such as Where_Used whose newlines are meaningful record
    separators. Collapses runs of spaces/tabs within each line and trims each
    line, but keeps the newline structure so downstream splitting still works.
    """
    if is_null(v):
        return np.nan
    s = str(v).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    s = "\n".join(lines).strip()
    if s.lower() in {t.lower() for t in null_tokens}:
        return np.nan
    return s if s != "" else np.nan


def normalize_name(v, name_noise) -> str:
    """Lowercase, strip punctuation, drop noise + version tokens and bare ints.

    Used for join keys and duplicate matching so "2019 Annual Performance (Copy)"
    and "Annual Performance v2" collapse toward the same signature.
    """
    s = re.sub(r"[^a-z0-9 ]+", " ", text(v).lower())
    noise = set(name_noise)
    toks = [t for t in s.split() if t and t not in noise and not t.isdigit()]
    return " ".join(toks).strip()


# ---- table-level cleaning -------------------------------------------------
def clean_table(df: pd.DataFrame, field_types: dict[str, str], cfg_clean) -> pd.DataFrame:
    """Apply per-field-type coercion to a column-renamed (canonical) frame."""
    out = df.copy()
    null_tokens = cfg_clean["null_tokens"]
    for col in out.columns:
        ftype = field_types.get(col, "text")
        if ftype == "date":
            out[col] = out[col].map(to_date)
        elif ftype == "number":
            out[col] = out[col].map(to_number)
        elif ftype == "bool":
            out[col] = out[col].map(lambda v: to_bool(v, cfg_clean))
        elif ftype == "multiline":
            out[col] = out[col].map(lambda v: normalize_multiline(v, null_tokens))
        else:
            out[col] = out[col].map(lambda v: normalize_text(v, null_tokens))
    return out
