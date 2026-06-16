"""Read .csv / .xlsx / .xls into a DataFrame of strings.

Everything is read as text first; clean.py coerces types later so pandas never
guesses (and mangles) IDs, dates, or booleans during ingest.

Two real-world robustness steps happen here so downstream column-mapping does not
silently fail (the classic "every report scores the same" bug):
  * CSV delimiter is sniffed (comma / semicolon / tab / pipe).
  * Leading title / filter rows above the real header row are detected and
    skipped (Workday exports often start with a report title and an "End Date"
    filter row before the actual column headers).
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from .security import check_table_size

# Scan at most this many leading rows when hunting for the real header row.
_HEADER_SCAN_ROWS = 15

# Prefer calamine (Rust) for .xlsx — it is several times faster than openpyxl on
# the large Fields export and produces byte-identical string output under
# dtype=str. Fall back to openpyxl when python-calamine isn't installed.
try:
    import python_calamine  # noqa: F401  (import is the availability probe)
    _XLSX_ENGINE = "calamine"
except ImportError:
    _XLSX_ENGINE = "openpyxl"


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


def _row_filled(row) -> int:
    """Count non-blank cells in a raw (header=None) row."""
    return sum(1 for v in row if v is not None and not pd.isna(v) and str(v).strip() != "")


def _detect_header_row(raw: pd.DataFrame) -> int:
    """Return the index of the real header row in a header-less frame.

    Title/filter rows above the headers (e.g. "All Custom Reports", then
    "End Date | 6/2/2026") have only a cell or two populated, whereas the header
    row is (nearly) full width. We take the FIRST row whose populated-cell count
    reaches a healthy fraction of the widest row seen — that is the header. A
    normal file with headers on row 0 returns 0 (nothing skipped).
    """
    scan = min(_HEADER_SCAN_ROWS, len(raw))
    if scan == 0:
        return 0
    counts = [_row_filled(raw.iloc[i]) for i in range(scan)]
    widest = max(counts)
    if widest <= 1:
        return 0
    threshold = max(3, int(0.6 * widest))
    for i, c in enumerate(counts):
        if c >= threshold:
            return i
    return 0


def _apply_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the detected header row to column names and drop everything above it."""
    hdr = _detect_header_row(raw)
    names = []
    for i, v in enumerate(raw.iloc[hdr].tolist()):
        name = "" if (v is None or pd.isna(v)) else str(v).strip()
        names.append(name if name else f"Unnamed: {i}")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = names
    return df.reset_index(drop=True)


def _read_csv_ragged(path: Path, sep: str) -> pd.DataFrame:
    """Read a CSV into a header-less, rectangular all-string frame.

    Parsing with the stdlib csv reader (rather than pandas) means a short
    title/filter row above the headers cannot fix the column count or raise a
    tokenizing error — rows are padded to the widest row. Quoted multiline cells
    (e.g. the Fields export's Where_Used) are preserved. utf-8-sig drops a BOM.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=sep))
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    padded = [
        [(c if (c is not None and c.strip() != "") else None) for c in r] + [None] * (width - len(r))
        for r in rows
    ]
    return pd.DataFrame(padded)


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    # Read with NO header first so we can locate the real header row ourselves.
    if ext == ".csv":
        # utf-8-sig drops a leading BOM; the sniffed sep handles non-comma exports.
        raw = _read_csv_ragged(path, _sniff_delimiter(path))
    elif ext == ".xlsx":
        raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str, engine=_XLSX_ENGINE)
    elif ext == ".xls":
        try:
            raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str, engine="xlrd")
        except ImportError as exc:  # xlrd not installed
            raise UnsupportedFormat(
                ".xls requires xlrd==1.2.0 (pip install 'xlrd==1.2.0'), "
                "or re-save the file as .xlsx."
            ) from exc
    else:
        raise UnsupportedFormat(f"Unsupported file type: {ext}")

    df = _apply_header(raw)
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
