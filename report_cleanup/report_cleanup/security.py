"""Centralized security helpers shared across the engine, CLI, and web layers.

Everything here is deliberately small and dependency-light so the protections
stay easy to audit:

  * secure_mkdir         — create output dirs with owner-only perms (POSIX)
  * check_table_size     — reject oversized parsed input (rows/cols)
  * excel_safe / sanitize_df_for_excel — neutralize Excel formula injection
  * sensitive_mode_enabled / mask_*    — optional masking of sensitive columns
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pandas.api import types as _pdt

# ---------------------------------------------------------------------------
# Parse-size limits
# ---------------------------------------------------------------------------
# [SECURITY] Cap the number of rows/columns we accept *after* parsing. The
# Flask MAX_CONTENT_LENGTH guards the raw upload size, but a small file can
# still expand into a huge frame (e.g. a CSV bomb). Defaults are generous for
# real Workday catalogs and overridable via env for unusual deployments.
MAX_INPUT_ROWS = int(os.environ.get("REPORT_CLEANUP_MAX_ROWS", "1000000"))
MAX_INPUT_COLS = int(os.environ.get("REPORT_CLEANUP_MAX_COLS", "1000"))


def check_table_size(df: pd.DataFrame, label: str) -> None:
    """Raise ValueError if a parsed table exceeds the configured limits."""
    rows = len(df)
    cols = df.shape[1] if df.ndim == 2 else 0
    if rows > MAX_INPUT_ROWS:
        raise ValueError(
            f"{label}: {rows:,} rows exceeds the limit of {MAX_INPUT_ROWS:,}."
        )
    if cols > MAX_INPUT_COLS:
        raise ValueError(
            f"{label}: {cols:,} columns exceeds the limit of {MAX_INPUT_COLS:,}."
        )


# ---------------------------------------------------------------------------
# Output directory hardening
# ---------------------------------------------------------------------------
def secure_mkdir(path: str | Path) -> Path:
    """Create a directory and, on POSIX, restrict it to the owner (0o700).

    Output dirs hold generated workbooks and the SQLite DB, both of which can
    contain sensitive HR data. chmod is skipped on non-POSIX platforms and any
    OSError (unsupported filesystem) is swallowed so the run never breaks.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(p, 0o700)
        except OSError:
            pass
    return p


# ---------------------------------------------------------------------------
# Excel formula-injection sanitization
# ---------------------------------------------------------------------------
# [SECURITY] Values that begin with any of these are treated as formulas by
# Excel/Sheets. A malicious report name like "=HYPERLINK(...)" or "=cmd|..."
# could exfiltrate data or run on a reviewer's machine. Prefixing with an
# apostrophe forces the cell to render as literal text.
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def excel_safe(value):
    """Prefix formula-leading strings with an apostrophe; pass others through."""
    if isinstance(value, str) and value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def sanitize_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every text cell formula-injection-safe."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        # Only touch text columns; numeric/date columns can't carry formulas.
        # pandas 2.x may type string columns as StringDtype (not object), so we
        # check both rather than relying on `== object`.
        s = out[col]
        if _pdt.is_object_dtype(s) or _pdt.is_string_dtype(s):
            out[col] = s.map(excel_safe)
    return out


# ---------------------------------------------------------------------------
# Sensitive mode / column masking
# ---------------------------------------------------------------------------
# Display-layer column names that identify people. Masked only in previews and
# Excel output — the underlying records are untouched so scoring/dedup logic
# that depends on these fields keeps working.
SENSITIVE_COLUMNS = {
    "Owner",
    "Created By",
    "Last Run By",
    "Requested By Name",
    "Requested By Employee ID",
}
_MASK = "***"


def sensitive_mode_enabled() -> bool:
    """True when REPORT_CLEANUP_SENSITIVE_MODE is set to a truthy value.

    Read live (not cached) so the CLI --sensitive flag can set the env var at
    startup and have it take effect for the rest of the process.
    """
    return os.environ.get("REPORT_CLEANUP_SENSITIVE_MODE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def mask_value(v):
    """Replace any non-empty value with a redaction token; keep blanks blank."""
    s = "" if v is None else str(v)
    return _MASK if s.strip() else s


def mask_df(df: pd.DataFrame, columns=SENSITIVE_COLUMNS) -> pd.DataFrame:
    """Return a copy of df with sensitive-named columns masked."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if col in columns:
            out[col] = out[col].map(mask_value)
    return out


def mask_record(d: dict, columns=SENSITIVE_COLUMNS) -> dict:
    """Return a copy of a row dict with sensitive keys masked."""
    return {k: (mask_value(v) if k in columns else v) for k, v in d.items()}
