"""Read .csv / .xlsx / .xls into a DataFrame of strings.

Everything is read as text first; clean.py coerces types later so pandas never
guesses (and mangles) IDs, dates, or booleans during ingest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .security import check_table_size


class UnsupportedFormat(Exception):
    pass


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
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
