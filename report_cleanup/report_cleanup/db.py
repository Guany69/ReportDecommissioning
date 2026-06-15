"""SQLite persistence of all run artifacts (reports, reasons, dup groups, field data)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .clean import normalize_report_name, text
from .security import secure_mkdir

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT, finished_at TEXT,
  table1_file TEXT, table1_sha TEXT, table1_rows INTEGER,
  table2_file TEXT, table2_sha TEXT, table2_rows INTEGER,
  table3_file TEXT, table3_sha TEXT, table3_rows INTEGER,
  config_snapshot TEXT,
  join_composite INTEGER, join_name_only INTEGER, join_unmatched INTEGER, join_ambiguous INTEGER
);
CREATE TABLE IF NOT EXISTS reports (
  report_uid INTEGER, run_id INTEGER,
  report_name TEXT, report_type TEXT, owner TEXT, created_by TEXT,
  report_tag TEXT, category TEXT, data_source TEXT,
  landing_page TEXT, areas_used TEXT, report_prompts TEXT, description TEXT,
  report_fields TEXT, worklet TEXT, shared TEXT, chart_type TEXT,
  created_date TEXT, last_updated TEXT, last_run_date TEXT, last_run_by TEXT,
  times_run REAL,
  exec_count INTEGER, last_exec_date TEXT, distinct_requesters INTEGER,
  appears_in_exec INTEGER, eff_last_run TEXT,
  comprehensive_last_run_date TEXT, effective_last_run_date TEXT,
  comprehensive_exec_count REAL, runs_exec_count INTEGER,
  is_hard_rule INTEGER, hard_rule_id TEXT, hard_rule_triggered INTEGER,
  hard_rule_name TEXT, hard_rule_reason TEXT,
  usage_risk INTEGER, age_risk INTEGER, usage_context_risk INTEGER,
  cleanup_risk_points INTEGER, cleanup_percentage INTEGER,
  business_protection_credit INTEGER, overall_score INTEGER,
  recurrence_classification TEXT, recurrence_cadence TEXT, recurrence_match_percentage REAL,
  potential_duplicate INTEGER, potential_duplicate_of TEXT,
  duplicate_similarity REAL, duplicate_relationship TEXT,
  field_jaccard_similarity REAL, smaller_report_containment REAL,
  total_risk_score INTEGER, recommendation TEXT,
  ownership_flags TEXT, data_quality_flags TEXT, all_flags TEXT,
  dup_group_id TEXT, is_suggested_keeper INTEGER, suggested_keeper_report_name TEXT,
  candidate_similarity_score REAL, report_name_similarity_percent REAL,
  field_count INTEGER, shared_field_count INTEGER,
  field_similarity_percent REAL, field_containment_percent REAL,
  missing_fields_compared_to_keeper TEXT, extra_fields_in_keeper TEXT,
  duplicate_classification TEXT, suggested_action TEXT, consolidation_reason TEXT,
  field_extraction_status TEXT, field_signature TEXT,
  PRIMARY KEY (report_uid, run_id)
);
CREATE TABLE IF NOT EXISTS report_reasons (
  reason_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, report_uid INTEGER,
  category TEXT, label TEXT, points INTEGER
);
CREATE TABLE IF NOT EXISTS dup_groups (
  dup_group_id TEXT, run_id INTEGER,
  member_count INTEGER, keeper_uid INTEGER, detection TEXT,
  PRIMARY KEY (dup_group_id, run_id)
);
CREATE TABLE IF NOT EXISTS report_fields (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  report_uid INTEGER,
  report_key TEXT,
  field_key TEXT,
  field_name TEXT,
  business_object TEXT,
  related_business_object TEXT,
  report_field_type TEXT,
  domain TEXT,
  categories TEXT,
  authorized_usage TEXT,
  source_match_type TEXT
);
CREATE TABLE IF NOT EXISTS field_rollups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  report_key TEXT,
  field_count INTEGER,
  field_signature TEXT,
  field_list_text TEXT,
  business_objects_used TEXT,
  domains_used TEXT,
  categories_used TEXT
);
CREATE TABLE IF NOT EXISTS field_diagnostics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  metric_name TEXT,
  metric_value TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(run_id);
CREATE INDEX IF NOT EXISTS idx_reasons_uid ON report_reasons(report_uid, run_id);
CREATE INDEX IF NOT EXISTS idx_reports_dup ON reports(dup_group_id);
CREATE INDEX IF NOT EXISTS idx_rfields_run ON report_fields(run_id);
CREATE INDEX IF NOT EXISTS idx_rollups_run ON field_rollups(run_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    # [SECURITY] Owner-only perms on the directory holding the SQLite DB,
    # which stores sensitive HR metadata.
    secure_mkdir(Path(db_path).parent)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def _fmt(v):
    """Render a value for SQLite: dates -> ISO string, lists -> '; ', NaN/NaT -> None."""
    if v is None:
        return None
    if isinstance(v, (list, set, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return int(v)
    return v


def create_run(conn, meta: dict) -> int:
    cur = conn.execute(
        """INSERT INTO runs (started_at, finished_at, table1_file, table1_sha, table1_rows,
            table2_file, table2_sha, table2_rows, table3_file, table3_sha, table3_rows,
            config_snapshot, join_composite, join_name_only, join_unmatched, join_ambiguous)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (meta.get("started_at"), meta.get("finished_at"),
         meta.get("table1_file"), meta.get("table1_sha"), meta.get("table1_rows"),
         meta.get("table2_file"), meta.get("table2_sha"), meta.get("table2_rows"),
         meta.get("table3_file"), meta.get("table3_sha"), meta.get("table3_rows"),
         meta.get("config_snapshot"),
         meta["diag"].get("composite"), meta["diag"].get("name_only"),
         meta["diag"].get("unmatched"), meta["diag"].get("ambiguous")),
    )
    conn.commit()
    return cur.lastrowid


REPORT_COLS = [
    "report_uid", "report_name", "report_type", "owner", "created_by", "report_tag",
    "category", "data_source", "landing_page", "areas_used", "report_prompts",
    "description", "report_fields", "worklet", "shared", "chart_type", "created_date",
    "last_updated", "last_run_date", "last_run_by", "times_run", "exec_count",
    "last_exec_date", "distinct_requesters", "appears_in_exec", "eff_last_run",
    "comprehensive_last_run_date", "effective_last_run_date",
    "comprehensive_exec_count", "runs_exec_count",
    "is_hard_rule", "hard_rule_id", "hard_rule_triggered", "hard_rule_name", "hard_rule_reason",
    "usage_risk", "age_risk", "usage_context_risk",
    "cleanup_risk_points", "cleanup_percentage", "business_protection_credit", "overall_score",
    "recurrence_classification", "recurrence_cadence", "recurrence_match_percentage",
    "potential_duplicate", "potential_duplicate_of",
    "duplicate_similarity", "duplicate_relationship",
    "field_jaccard_similarity", "smaller_report_containment",
    "total_risk_score", "recommendation",
    "ownership_flags", "data_quality_flags", "all_flags",
    "dup_group_id", "is_suggested_keeper", "suggested_keeper_report_name",
    "candidate_similarity_score", "report_name_similarity_percent",
    "field_count", "shared_field_count",
    "field_similarity_percent", "field_containment_percent",
    "missing_fields_compared_to_keeper", "extra_fields_in_keeper",
    "duplicate_classification", "suggested_action", "consolidation_reason",
    "field_extraction_status", "field_signature",
]


def write_reports(conn, run_id: int, records: list[dict]) -> None:
    rows = []
    for r in records:
        rows.append([run_id] + [_fmt(r.get(c)) for c in REPORT_COLS])
    placeholders = ",".join(["?"] * (len(REPORT_COLS) + 1))
    conn.executemany(
        f"INSERT INTO reports (run_id, {', '.join(REPORT_COLS)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()


def write_reasons(conn, run_id: int, records: list[dict]) -> None:
    rows = []
    for r in records:
        for reason in r.get("all_reasons", []):
            rows.append((run_id, r["report_uid"], reason.category, reason.label, reason.points))
    conn.executemany(
        "INSERT INTO report_reasons (run_id, report_uid, category, label, points) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()


def write_groups(conn, run_id: int, groups) -> None:
    rows = [(g.group_id, run_id, len(g.members), g.keeper_uid, g.detection) for g in groups]
    conn.executemany(
        "INSERT INTO dup_groups (dup_group_id, run_id, member_count, keeper_uid, detection) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()


def write_field_data(conn, run_id: int, rollup_result: dict, records: list[dict], cfg) -> None:
    """Write report_fields (individual rows), field_rollups, and field_diagnostics."""
    if not rollup_result:
        return

    # Build report_uid lookup by EXACT normalized name key (mirrors attach_report_fields).
    uid_by_key: dict[str, int] = {}
    for r in records:
        rk = normalize_report_name(r.get("report_name", ""))
        if rk:
            uid_by_key[rk] = r["report_uid"]

    # Write individual field rows.
    field_rows: list[tuple] = []
    for row in rollup_result.get("individual_rows", []):
        rk = row.get("report_key", "")
        uid = uid_by_key.get(rk)  # None when the rollup key had no matching report
        field_rows.append((
            run_id, uid, rk,
            row.get("field_key"), row.get("field_name"),
            row.get("business_object"), row.get("related_business_object"),
            row.get("report_field_type"), row.get("domain"),
            row.get("categories"), row.get("authorized_usage"),
            row.get("source_match_type"),
        ))
    if field_rows:
        conn.executemany(
            """INSERT INTO report_fields
               (run_id, report_uid, report_key, field_key, field_name,
                business_object, related_business_object, report_field_type,
                domain, categories, authorized_usage, source_match_type)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            field_rows,
        )

    # Write rollup summaries.
    rollup = rollup_result.get("rollup", {})
    rollup_rows: list[tuple] = []
    for rk, entry in rollup.items():
        rollup_rows.append((
            run_id, rk,
            entry.get("field_count", 0),
            entry.get("field_signature"),
            entry.get("field_list_text"),
            "; ".join(entry.get("business_objects_used") or []),
            "; ".join(entry.get("domains_used") or []),
            "; ".join(entry.get("categories_used") or []),
        ))
    if rollup_rows:
        conn.executemany(
            """INSERT INTO field_rollups
               (run_id, report_key, field_count, field_signature, field_list_text,
                business_objects_used, domains_used, categories_used)
               VALUES (?,?,?,?,?,?,?,?)""",
            rollup_rows,
        )

    # Write diagnostics (key-value).
    diag = rollup_result.get("diagnostics", {})
    diag_rows = [(run_id, k, str(v)) for k, v in diag.items()]
    if diag_rows:
        conn.executemany(
            "INSERT INTO field_diagnostics (run_id, metric_name, metric_value) VALUES (?,?,?)",
            diag_rows,
        )

    conn.commit()


def reset(conn) -> None:
    """Truncate-and-load: single active run (keep_history=false).

    Drops and recreates tables so the live schema always matches the code.
    """
    for t in ("report_reasons", "reports", "dup_groups", "runs",
              "report_fields", "field_rollups", "field_diagnostics"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(SCHEMA)
    conn.commit()
