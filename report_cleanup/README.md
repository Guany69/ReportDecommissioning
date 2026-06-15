# Report Decommissioning & Consolidation System (MVP)

Local, offline engine that ingests two report exports (metadata + execution),
joins them, applies hard + soft deletion logic, finds duplicate clusters, ranks
each cluster by comprehensiveness, and emits a staff-reviewable Excel workbook
plus an optional Streamlit dashboard. Nothing is auto-deleted — every
recommendation carries a stored reason trail.

See the full design in `../../.claude/plans/` or the sections referenced below.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Only if you must read legacy .xls files:
# pip install "xlrd==1.2.0"
```

## Quick start

```bash
# 1. Generate synthetic sample exports (uses your exact source field names)
python sample_data/generate_sample.py

# 2. Run the pipeline
python -m report_cleanup.run \
    --table1 sample_data/metadata.csv \
    --table2 sample_data/execution.csv \
    --out ./output

# 3. (optional) Review in the dashboard
streamlit run app.py
```

Outputs land in `./output/`:
- `report-cleanup-<date>.xlsx` — 10-tab workbook for staff.
- `report_cleanup.db` — SQLite with `reports`, `report_reasons`, `dup_groups`, `runs`.

## Using your real data

The source headers are mapped in `config.yaml` under `aliases`. The defaults
already match the supplied field names (e.g. `Custom Report`, `Number of times`,
`Worklet Landing Pages`, `Areas where used`). Drop your real exports in and run —
no code changes needed. Tune weights / thresholds / score bands in `config.yaml`.

## Tests

```bash
pytest -q
```

## Module map

| File | Responsibility |
|---|---|
| `config.yaml` / `config.py` | All tunables; the engine reads only from here |
| `schema.py` | Canonical fields + header auto-mapping |
| `io_readers.py` | Read csv/xls/xlsx |
| `validate.py` | Required-field + data-quality checks |
| `clean.py` | Type coercion + null/bool/name helpers |
| `exec_rollup.py` | Table 2 -> per-report execution rollup |
| `join.py` | Composite (name+type+owner) join with name-only fallback |
| `hard_rules.py` | DNU, orphan worklet |
| `soft_scoring.py` | Tiered/capped weighted scoring + reasons |
| `dedup.py` | Exact + RapidFuzz fuzzy clustering (union-find) |
| `comprehensiveness.py` | Per-group member scoring |
| `recommend.py` | Keeper selection + final recommendation |
| `db.py` | SQLite persistence |
| `export_excel.py` | Multi-tab workbook |
| `pipeline.py` / `run.py` | Orchestration + CLI |
| `app.py` | Streamlit dashboard |
