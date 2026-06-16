"""Read .csv / .xlsx / .xls into a DataFrame of strings.

Everything is read as text first; clean.py coerces types later so pandas never
guesses (and mangles) IDs, dates, or booleans during ingest.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from .security import check_table_size


class UnsupportedFormat(Exception):
    pass


def _sniff_delimiter(path: Path) -> str:
    """Detect the CSV delimiter so semicolon/tab/pipe exports parse correctly.

    Workday CSV exports are frequently semicolon- or tab-delimited; read as
    comma they collapse into a single column and every scoring field reads as
    null. Sniff a sample (restricted to common delimiters) and fall back to a
    comma when detection is ambiguous.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(16384)
    if not sample:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # No clear delimiter (e.g. a genuine single-column file) — default to comma.
        return ","


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        # utf-8-sig drops a leading BOM; the sniffed sep handles non-comma exports.
        sep = _sniff_delimiter(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""],
                         sep=sep, encoding="utf-8-sig")
    elif ext == ".xlsx":
        df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl")
    elif ext == ".xls":
        try:
            df = pd.read_excel(path, sheet_name=0, dtype=str, engine="xlrd")
        except ImportError as exc:  # xlrd not installed
            raise UnsupportedFormat(
                ".xls requires xlrd==1.2.0 (pip install 'xlrd==1.2.0'), "
                "or re-save the file as .xlsx."
            ) from exc
    else:
        raise UnsupportedFormat(f"Unsupported file type: {ext}")
    # [SECURITY] Reject oversized parsed input (CSV/XLSX bombs) before the
    # rest of the pipeline allocates memory per row/column.
    check_table_size(df, path.name)
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
