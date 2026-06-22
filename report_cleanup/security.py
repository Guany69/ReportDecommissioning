"""Centralized security helpers shared across the engine, CLI, and web layers.

Everything here is deliberately small and dependency-light so the protections
stay easy to audit:

  * secure_mkdir         — create output dirs with owner-only perms (POSIX)
  * check_table_size     — reject oversized parsed input (rows/cols)
  * excel_safe / sanitize_df_for_excel — neutralize Excel formula injection
  * sensitive_mode_enabled / mask_*    — optional masking of sensitive columns
"""
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
from pandas.api import types as _pdt

logger = logging.getLogger(__name__)


class UnsafeUpload(Exception):
    """Raised when an uploaded file fails a pre-parse safety check."""


# ---------------------------------------------------------------------------
# Pre-parse upload limits (size, signature, decompression-bomb)
# ---------------------------------------------------------------------------
# [SECURITY] These are checked on the raw bytes *before* any parser allocates
# memory, so an oversized upload, a renamed binary, or an XLSX "zip bomb" is
# rejected up front instead of after it has already inflated in RAM. All limits
# are env-overridable for unusual (trusted, very large) deployments.
#
# Default raw-upload ceiling. The Streamlit server limit (.streamlit/config.toml
# server.maxUploadSize) is the first line of defense for the web UI; this is the
# engine-level backstop that also covers the CLI.
MAX_UPLOAD_BYTES = int(os.environ.get("REPORT_CLEANUP_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
# Cap the *uncompressed* size of an .xlsx (a zip) and the inflation ratio so a
# small file cannot decompress into gigabytes.
MAX_XLSX_UNCOMPRESSED_BYTES = int(
    os.environ.get("REPORT_CLEANUP_MAX_XLSX_UNCOMPRESSED_BYTES", str(500 * 1024 * 1024))
)
MAX_XLSX_COMPRESSION_RATIO = float(os.environ.get("REPORT_CLEANUP_MAX_XLSX_RATIO", "200"))

# File-type signatures (magic bytes). Extension alone is not trusted.
_XLSX_MAGIC = b"PK\x03\x04"                       # .xlsx is a ZIP container
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # .xls is an OLE2 compound file


def validate_upload(path: str | Path, ext: str | None = None) -> None:
    """Reject an upload before parsing if it is too large or not what it claims.

    [SECURITY] Guards findings around extension-only validation and parser DoS:
      * caps the raw byte size (oversized-upload / temp-disk exhaustion),
      * verifies the leading magic bytes match the declared extension (a renamed
        executable or mismatched container is refused, not handed to a parser),
      * for .xlsx, inspects the ZIP directory for decompression-bomb traits.
    Raises UnsafeUpload on any violation.
    """
    p = Path(path)
    ext = (ext or p.suffix).lower()

    size = p.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise UnsafeUpload(
            f"{p.name}: {size:,} bytes exceeds the upload limit of {MAX_UPLOAD_BYTES:,} bytes."
        )

    with open(p, "rb") as fh:
        head = fh.read(8192)

    if ext == ".xlsx":
        if not head.startswith(_XLSX_MAGIC):
            raise UnsafeUpload(f"{p.name}: not a valid .xlsx file (bad signature).")
        _check_zip_bomb(p)
    elif ext == ".xls":
        if not head.startswith(_OLE_MAGIC):
            raise UnsafeUpload(f"{p.name}: not a valid legacy .xls file (bad signature).")
    elif ext == ".csv":
        # A NUL byte in the leading sample means this is binary content, not a
        # text/CSV export (e.g. a renamed executable or spreadsheet).
        if b"\x00" in head:
            raise UnsafeUpload(f"{p.name}: looks binary, not a text CSV.")
    else:
        raise UnsafeUpload(f"{p.name}: unsupported file type: {ext}")


def _check_zip_bomb(path: Path) -> None:
    """Reject an .xlsx whose ZIP directory advertises bomb-like inflation."""
    try:
        with zipfile.ZipFile(path) as zf:
            total_uncompressed = 0
            total_compressed = 0
            for info in zf.infolist():
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
    except zipfile.BadZipFile as exc:
        raise UnsafeUpload(f"{path.name}: corrupt or malformed .xlsx container.") from exc

    if total_uncompressed > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise UnsafeUpload(
            f"{path.name}: uncompressed size {total_uncompressed:,} bytes exceeds the "
            f"limit of {MAX_XLSX_UNCOMPRESSED_BYTES:,} bytes (possible zip bomb)."
        )
    # Only flag a high ratio once the payload is non-trivial, so ordinary small
    # well-compressed sheets aren't rejected.
    if (
        total_compressed > 0
        and total_uncompressed > 10 * 1024 * 1024
        and total_uncompressed / total_compressed > MAX_XLSX_COMPRESSION_RATIO
    ):
        raise UnsafeUpload(
            f"{path.name}: compression ratio "
            f"{total_uncompressed / total_compressed:.0f}x exceeds the limit "
            f"of {MAX_XLSX_COMPRESSION_RATIO:.0f}x (possible zip bomb)."
        )


# ---------------------------------------------------------------------------
# Parse-size limits
# ---------------------------------------------------------------------------
# [SECURITY] Cap the number of rows/columns we accept *after* parsing. The
# pre-parse upload limit guards the raw bytes, but a small file can still expand
# into a huge frame (e.g. a CSV bomb). Defaults are generous for real Workday
# catalogs and overridable via env for unusual deployments.
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
        except OSError as exc:
            # [SECURITY] Do not fail the run, but never fail *silently*: an
            # operator must be able to tell that owner-only perms were NOT
            # applied (fail-open visibility) rather than assuming they were.
            logger.warning(
                "Could not set owner-only (0o700) permissions on %s: %s. "
                "The directory may be readable by other local users.", p, exc
            )
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

# The same person-identifying fields keyed by their lowercase *record/DB* names
# (see db.REPORT_COLS). Used to mask rows before they are written to SQLite when
# sensitive mode is on, so the database is not a plaintext bypass of the preview
# and Excel masking.
SENSITIVE_RECORD_KEYS = {
    "owner",
    "created_by",
    "last_run_by",
    "requested_by_name",
    "requested_by_employee_id",
}
_MASK = "***"


def auth_required() -> bool:
    """True when REPORT_CLEANUP_REQUIRE_AUTH is set to a truthy value.

    When set, the Streamlit app refuses to serve any data until the visitor
    authenticates (native OIDC login if configured, otherwise a shared access
    code). Off by default so trusted local/offline CLI use is unchanged.
    """
    return os.environ.get("REPORT_CLEANUP_REQUIRE_AUTH", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


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
