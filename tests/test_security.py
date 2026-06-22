"""Security-focused tests for the hardening helpers and pipeline behavior."""
from __future__ import annotations

import os
import sqlite3
import stat

import pandas as pd
import pytest

from report_cleanup import security
from report_cleanup.pipeline import run_pipeline


# --------------------------------------------------------------------------
# Excel formula-injection sanitization
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["=1+1", "+1", "-1", "@SUM(A1)", "=HYPERLINK('http://x')"])
def test_excel_safe_prefixes_formula_strings(raw):
    out = security.excel_safe(raw)
    assert out == "'" + raw


@pytest.mark.parametrize("raw", ["Report A", "owner@example.com is fine mid-string", "", "123"])
def test_excel_safe_leaves_normal_strings(raw):
    assert security.excel_safe(raw) == raw


def test_excel_safe_ignores_non_strings():
    assert security.excel_safe(5) == 5
    assert security.excel_safe(-5) == -5  # numeric, not a formula string
    assert security.excel_safe(None) is None


def test_sanitize_df_for_excel_neutralizes_cells():
    df = pd.DataFrame({"Report Name": ["=cmd|'/c calc'!A1", "normal"], "Count": [1, 2]})
    out = security.sanitize_df_for_excel(df)
    assert out["Report Name"][0].startswith("'=")
    assert out["Report Name"][1] == "normal"
    assert list(out["Count"]) == [1, 2]  # numeric column untouched


# --------------------------------------------------------------------------
# Parse-size limits
# --------------------------------------------------------------------------
def test_check_table_size_rejects_too_many_rows(monkeypatch):
    monkeypatch.setattr(security, "MAX_INPUT_ROWS", 2)
    with pytest.raises(ValueError, match="rows exceeds"):
        security.check_table_size(pd.DataFrame({"a": [1, 2, 3]}), "t1")


def test_check_table_size_rejects_too_many_cols(monkeypatch):
    monkeypatch.setattr(security, "MAX_INPUT_COLS", 1)
    with pytest.raises(ValueError, match="columns exceeds"):
        security.check_table_size(pd.DataFrame({"a": [1], "b": [2]}), "t1")


def test_check_table_size_allows_within_limits():
    security.check_table_size(pd.DataFrame({"a": [1, 2]}), "t1")  # no raise


# --------------------------------------------------------------------------
# Output directory hardening
# --------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="POSIX permission check")
def test_secure_mkdir_is_owner_only(tmp_path):
    d = security.secure_mkdir(tmp_path / "out")
    mode = stat.S_IMODE(os.stat(d).st_mode)
    assert mode == 0o700


# --------------------------------------------------------------------------
# Sensitive mode / masking
# --------------------------------------------------------------------------
def test_sensitive_mode_reads_env(monkeypatch):
    monkeypatch.delenv("REPORT_CLEANUP_SENSITIVE_MODE", raising=False)
    assert security.sensitive_mode_enabled() is False
    monkeypatch.setenv("REPORT_CLEANUP_SENSITIVE_MODE", "1")
    assert security.sensitive_mode_enabled() is True
    monkeypatch.setenv("REPORT_CLEANUP_SENSITIVE_MODE", "off")
    assert security.sensitive_mode_enabled() is False


def test_mask_df_only_touches_sensitive_columns():
    df = pd.DataFrame({"Owner": ["Alice", ""], "Report Name": ["R1", "R2"]})
    out = security.mask_df(df)
    assert list(out["Owner"]) == ["***", ""]          # non-empty masked, blank kept
    assert list(out["Report Name"]) == ["R1", "R2"]   # non-sensitive untouched


def test_mask_record_masks_keys():
    rec = {"Owner": "Bob", "Created By": "Carol", "Report Name": "R1"}
    out = security.mask_record(rec)
    assert out == {"Owner": "***", "Created By": "***", "Report Name": "R1"}


# --------------------------------------------------------------------------
# Pipeline: no absolute/temp source paths persisted (item 8)
# --------------------------------------------------------------------------
def test_pipeline_stores_basename_not_full_path(tmp_path):
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "Custom Report,Number of Times Executed,Last Run Date\n"
        "Active Employee Report,240,2026-05-26\n"
    )
    execf = tmp_path / "execution.csv"
    execf.write_text(
        "Report Name,Start Date and Time,Requested By\n"
        "Active Employee Report,2026-05-26 09:00:00,Bob Lee\n"
    )
    res = run_pipeline(str(meta), str(execf), out_dir=str(tmp_path / "out"))

    conn = sqlite3.connect(res["db"])
    t1_file, t2_file = conn.execute("SELECT table1_file, table2_file FROM runs").fetchone()
    conn.close()
    assert t1_file == "metadata.csv"
    assert t2_file == "execution.csv"
    assert os.sep not in t1_file  # never a path
