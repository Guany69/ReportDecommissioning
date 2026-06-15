"""Orchestrates the full flow: ingest -> ... -> Excel + SQLite."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import db, schema
from .clean import clean_table
from .config import Config, load_config
from .dedup import CLASSIFICATION_META_ONLY, detect_duplicates
from .exec_rollup import build_exec_rollup
from .export_excel import export_workbook
from .field_rollup import (
    FIELD_EXPORT_MODE_MISSING,
    attach_report_fields,
    build_report_field_rollup,
    validate_field_table,
)
from .flags import build_all_flags
from .hard_rules import apply_hard_rules, has_recent_or_recurring_usage
from .io_readers import file_sha256, read_any
from .join import derive_effective_last_run, join_reports
from .overall import calculate_overall_score
from .protection import calculate_protection_credit
from .recurrence import detect_recurrence
from .duplicate_similarity import compute_duplicate_matches
from .recommend import (apply_duplicate_analysis, recommend_label,
                        resolve_recommendation)
from .soft_scoring import Reason, score_report
from .validate import validate


def _rename_to_canonical(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Keep only mapped columns, renamed source_header -> canonical_key."""
    inv = {src: key for key, src in mapping.items()}
    cols = [c for c in df.columns if c in inv]
    return df[cols].rename(columns=inv)


def run_pipeline(
    table1_path,
    table2_path=None,
    table3_fields_path=None,
    config_path=None,
    out_dir=None,
    config: Config | None = None,
) -> dict:
    cfg = config or load_config(config_path)
    out_dir = Path(out_dir) if out_dir else Path(cfg.get("run.excel_dir", "output"))
    started = datetime.now()

    # 1. INGEST
    t1_raw = read_any(table1_path)
    t2_raw = read_any(table2_path) if table2_path else pd.DataFrame()
    t3_raw = read_any(table3_fields_path) if table3_fields_path else pd.DataFrame()

    # 2. NORMALIZE COLUMNS (auto-map)
    t1_map = schema.auto_map(list(t1_raw.columns), cfg.aliases, schema.TABLE1_FIELDS)
    t2_map = (schema.auto_map(list(t2_raw.columns), cfg.aliases, schema.TABLE2_FIELDS)
              if len(t2_raw) else {})
    t3_map = (schema.auto_map(list(t3_raw.columns), cfg.aliases, schema.TABLE3_FIELDS)
              if len(t3_raw) else {})

    # 3. VALIDATE (table1 + table2)
    vr = validate(t1_raw, t2_raw if len(t2_raw) else None, t1_map, t2_map)
    if vr.has_fatal():
        raise ValueError("Validation failed:\n" + vr.summary())

    # 3b. VALIDATE field export — determines mode; never fatal
    fvr = validate_field_table(t3_raw if len(t3_raw) else None, t3_map)
    all_warnings = list(vr.warnings) + list(fvr.warnings)
    field_mode = fvr.mode

    # 4. CLEAN
    t1 = clean_table(_rename_to_canonical(t1_raw, t1_map), schema.FIELD_TYPES, cfg.clean)
    t2 = (clean_table(_rename_to_canonical(t2_raw, t2_map), schema.FIELD_TYPES, cfg.clean)
          if t2_map else pd.DataFrame())
    t3 = (clean_table(_rename_to_canonical(t3_raw, t3_map), schema.FIELD_TYPES, cfg.clean)
          if t3_map else pd.DataFrame())

    # 5. BUILD FIELD ROLLUP (table3 -> per-report field sets)
    if len(t3) and field_mode != FIELD_EXPORT_MODE_MISSING:
        field_rollup_result = build_report_field_rollup(t3, cfg)
    else:
        field_rollup_result = {"rollup": {}, "diagnostics": {}, "unmatched_rows": [], "individual_rows": []}

    # 6-7. EXEC ROLLUP + JOIN
    name_noise = cfg.clean["name_noise"]
    rollup = build_exec_rollup(t2, name_noise)
    records, diag = join_reports(t1, rollup, name_noise)
    records = derive_effective_last_run(records)
    by_uid = {r["report_uid"]: r for r in records}

    # 7b. RECURRENCE — inferred from Runs execution timestamps only.
    for r in records:
        rr = detect_recurrence(r.get("runs_start_timestamps", []), cfg).as_dict()
        r["recurrence"] = rr
        r["recurrence_classification"] = rr["classification"]
        r["recurrence_cadence"] = rr["cadence"]
        r["recurrence_match_percentage"] = rr["match_percentage"]

    # 8. ATTACH FIELD SETS — must happen before duplicate detection
    attach_report_fields(records, field_rollup_result, field_mode, cfg)

    # 9-10. SCORING: hard rules -> cleanup risk -> protection credit -> Overall Score.
    # Duplicate evidence is NOT an input here (it only affects wording later).
    for r in records:
        hit = apply_hard_rules(r, cfg)
        r["is_hard_rule"] = bool(hit)
        r["hard_rule_triggered"] = bool(hit)
        r["hard_rule_id"] = hit.rule_id if hit else None
        r["hard_rule_name"] = hit.name if hit else None
        r["hard_rule_reason"] = hit.reason if hit else None
        r["all_reasons"] = []
        if hit:
            r["all_reasons"].append(Reason("hard_rule", f"{hit.name}: {hit.reason}", None))

        # Cleanup risk (0..60) computed for every report for transparency.
        res = score_report(r, cfg)
        r["usage_risk"] = res.usage_risk
        r["age_risk"] = res.age_risk
        r["usage_context_risk"] = res.usage_context_risk
        r["cleanup_risk_points"] = res.total_risk_score
        r["total_risk_score"] = res.total_risk_score   # legacy alias (0..60)
        r["cleanup_reasons"] = list(res.reasons)

        # Business-protection credit (never applied to hard-rule rows).
        recurrence = r.get("recurrence")  # populated in Phase 2; None here
        if hit:
            credit, protection_reasons = 0, []
        else:
            credit, protection_reasons = calculate_protection_credit(r, recurrence, cfg)
        r["protection_reasons"] = protection_reasons

        # Overall Decommissioning Score (0..100; 100 reserved for hard rules).
        ov = calculate_overall_score(res.total_risk_score, credit, bool(hit), cfg)
        r["overall_score"] = ov.overall_score
        r["cleanup_risk_max"] = ov.cleanup_risk_max
        r["cleanup_percentage"] = ov.cleanup_percentage
        r["business_protection_credit"] = ov.business_protection_credit
        r["recommendation"] = recommend_label(ov.overall_score, cfg)

        # Assemble the full reason trail.
        r["all_reasons"].extend(res.reasons)
        r["all_reasons"].extend(protection_reasons)
        r["all_reasons"].extend(ov.reasons)

        # Conflict warning: hard-ruled but still actively used.
        if hit and has_recent_or_recurring_usage(r, cfg):
            warn = "Recent or recurring usage exists despite the hard-rule designation."
            r["all_reasons"].append(Reason("hard_rule", warn, None))
            r.setdefault("conflict_warnings", []).append(warn)

    # 11. OWNERSHIP / DATA-QUALITY FLAGS (no score weight)
    for r in records:
        r.update(build_all_flags(r, cfg))

    # 12. DUPLICATE DETECTION (two-stage) + KEEPER / per-member dup analysis
    groups, meta_only_pairs = detect_duplicates(records, cfg)
    apply_duplicate_analysis(by_uid, groups, cfg)

    # 12b. WEIGHTED DUPLICATE SIMILARITY (evidence only — never changes the score).
    compute_duplicate_matches(records, cfg)
    group_of = {}
    for g in groups:
        for u in g.members:
            group_of[u] = g

    # 13. ANNOTATE METADATA-ONLY PAIRS
    # Pairs that passed the Stage-1 gate but had no field evidence.
    # Only stamp the classification if the report is not already in a confirmed group.
    confirmed_uids = {uid for g in groups for uid in g.members}
    for uid_a, uid_b in meta_only_pairs:
        for uid in (uid_a, uid_b):
            r = by_uid.get(uid)
            if r is not None and uid not in confirmed_uids:
                if not r.get("duplicate_classification"):
                    r["duplicate_classification"] = CLASSIFICATION_META_ONLY

    # 14. SUGGESTED ACTION
    for r in records:
        g = group_of.get(r["report_uid"])
        r["suggested_action"] = resolve_recommendation(r, g, by_uid, cfg)
        r["all_reasons"].extend(r.get("dup_reasons", []))

    finished = datetime.now()

    # 15. BUILD FIELD DIAGNOSTICS
    total_reports = len(records)
    matched_count = sum(1 for r in records if r.get("field_extraction_status") == "Matched")
    field_diag = {
        "field_export_provided": bool(table3_fields_path),
        "field_export_mode": field_mode,
        "total_field_rows": field_rollup_result["diagnostics"].get("total_field_rows", 0),
        "total_report_field_links": field_rollup_result["diagnostics"].get("total_report_field_links", 0),
        "total_reports_with_fields": matched_count,
        "total_reports_without_fields": total_reports - matched_count,
        "total_unmatched_field_rows": field_rollup_result["diagnostics"].get("total_unmatched_rows", 0),
        "total_where_used_rows_parsed": field_rollup_result["diagnostics"].get("total_where_used_rows_parsed", 0),
        "total_where_used_parse_failures": field_rollup_result["diagnostics"].get("total_where_used_parse_failures", 0),
        "blank_where_used_rows": field_rollup_result["diagnostics"].get("blank_where_used_rows", 0),
        "multi_report_field_rows": field_rollup_result["diagnostics"].get("multi_report_field_rows", 0),
        "unmatched_where_used_reports": field_rollup_result["diagnostics"].get("unmatched_where_used_reports", 0),
        "reports_with_duplicate_field_keys": field_rollup_result["diagnostics"].get("reports_with_duplicate_field_keys", 0),
        "field_join_match_rate": (
            round(100.0 * matched_count / total_reports, 1) if total_reports else 0.0
        ),
        "metadata_only_pairs": len(meta_only_pairs),
    }

    # 16. PERSIST
    run_meta = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        # [SECURITY] Store the bare filename only — never the absolute/temp
        # source path. Full paths leak local usernames, directory layout, and
        # randomized temp-upload names into the SQLite DB and Excel snapshot.
        "table1_file": Path(table1_path).name,
        "table1_sha": file_sha256(table1_path),
        "table1_rows": len(t1_raw),
        "table2_file": Path(table2_path).name if table2_path else None,
        "table2_sha": file_sha256(table2_path) if table2_path else None,
        "table2_rows": len(t2_raw),
        "table3_file": Path(table3_fields_path).name if table3_fields_path else None,
        "table3_sha": file_sha256(table3_fields_path) if table3_fields_path else None,
        "table3_rows": len(t3_raw),
        "config_snapshot": cfg.snapshot_json(),
        "diag": diag,
        "field_diag": field_diag,
        "field_coverage": vr.field_coverage,
        "warnings": all_warnings,
    }

    conn = db.connect(out_dir / Path(cfg.get("run.db_path", "report_cleanup.db")).name)
    if not cfg.get("run.keep_history", False):
        db.reset(conn)
    run_id = db.create_run(conn, run_meta)
    db.write_reports(conn, run_id, records)
    db.write_reasons(conn, run_id, records)
    db.write_groups(conn, run_id, groups)
    db.write_field_data(conn, run_id, field_rollup_result, records, cfg)
    conn.close()

    # 17. EXPORT
    xlsx = export_workbook(records, groups, run_meta, out_dir, field_rollup_result)

    return {
        "run_id": run_id,
        "records": records,
        "groups": groups,
        "diag": diag,
        "field_diag": field_diag,
        "warnings": all_warnings,
        "xlsx": xlsx,
        "db": str(out_dir / Path(cfg.get("run.db_path", "report_cleanup.db")).name),
    }
