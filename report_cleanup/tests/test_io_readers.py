"""CSV reading must auto-detect the delimiter so non-comma exports parse into
columns (a semicolon/tab file read as comma collapses to one column, which makes
every report score the same)."""
from report_cleanup.io_readers import read_any

HEADER = ["Custom Report", "Number of Times Executed", "Last Run Date", "Shared"]
ROW = ["Active Employee Report", "240", "2026-05-01", "Yes"]


def _write(tmp_path, sep, name="t.csv", bom=False):
    p = tmp_path / name
    text = sep.join(HEADER) + "\n" + sep.join(ROW) + "\n"
    p.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return p


def test_comma_delimited(tmp_path):
    df = read_any(_write(tmp_path, ","))
    assert list(df.columns) == HEADER


def test_semicolon_delimited(tmp_path):
    df = read_any(_write(tmp_path, ";"))
    assert list(df.columns) == HEADER  # not one giant column


def test_tab_delimited(tmp_path):
    df = read_any(_write(tmp_path, "\t"))
    assert list(df.columns) == HEADER


def test_leading_bom_stripped(tmp_path):
    df = read_any(_write(tmp_path, ",", bom=True))
    # The first column header must not carry a BOM prefix.
    assert df.columns[0] == "Custom Report"
