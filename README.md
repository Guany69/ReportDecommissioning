# Report Decommissioning & Consolidation System

Local, offline engine that ingests three Workday exports (Comprehensive metadata,
Runs execution history, Fields/Where_Used), joins them on an exact report-name
key, and produces a single **Overall Decommissioning Score (0–100)** per report
with a complete, reproducible reason trail. It also detects recurring execution,
finds weighted duplicate relationships, and emits a staff-reviewable Excel
workbook, a SQLite snapshot, and an optional Streamlit dashboard. **Nothing is
ever auto-deleted** — every output is a recommendation.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Legacy binary `.xls` support comes from `xlrd>=2.0.1` (already in
`requirements.txt`). Modern xlrd is the maintained line and still reads `.xls`;
it only dropped the old `.xlsx` path, which we never use (calamine/openpyxl read
`.xlsx`). Re-save `.xls` files as `.xlsx` when you can — it is the faster path.

## Quick start

```bash
# 1. Run the pipeline against your Workday exports (table3 fields are
#    optional but enable duplicate analysis)
python -m report_cleanup.run \
    --table1 metadata.xlsx \
    --table2 execution.csv \
    --table3-fields fields.xlsx \
    --out ./output

# 2. (optional) Review in the dashboard
streamlit run app.py
```

Outputs land in `./output/`:
- `report-cleanup-<date>.xlsx` — multi-tab workbook (All Reports, Hard Rule
  Matches, one tab per recommendation tier, Duplicate Groups, Reason Detail,
  Field Rollup/Diagnostics, Data Quality, Config Snapshot).
- `report_cleanup.db` — SQLite with `reports`, `report_reasons`, `dup_groups`,
  `report_fields`, `field_rollups`, `runs`.

## The three inputs

1. **Comprehensive** (one row per report) — `Custom Report`, `Number of Times
   Executed`, `Last Run Date`, `Worklet Landing Pages`, `Landing Page`,
   `Area Where Used`, `Shared`, `Worklet`, … Headers are mapped in `config.yaml`
   under `aliases`.
2. **Runs** (one row per execution; ~6 month window) — `Report Name`,
   `Start Date and Time`, `Last Run Date`, `Requested By`, …
3. **Fields** (one row per unique field) — `Business Object`, `Field Name`,
   `Built-in Prompts`, `Authorized Usage`, and `Where_Used` (the reports that use
   the field, **one per line**).

Comprehensive and Runs are joined on an **exact, case-insensitive,
whitespace-trimmed** report name (`clean.normalize_report_name`) — never fuzzy.
Duplicate normalized Comprehensive names are surfaced as a data-quality warning.

## Scoring model

```
Overall Decommissioning Score (0..100)

  if hard rule (DNU in name / orphaned worklet):   score = 100
  else:
      cleanup_percentage = cleanup_risk_points / 60 * 100   # 0..60 rubric
      score = round(cleanup_percentage - business_protection_credit)
      score = clamp(score, 0, 99)                           # 100 reserved for hard rules
```

- **Cleanup risk (0–60)** — usage recency (using the **effective last-run date** =
  max of the Comprehensive date and the latest Runs execution), run count, age,
  and usage-context, all tiered/capped per `config.yaml`.
- **Business-protection credits (cap 20)** — recurrence strength, distinct
  requester breadth, and business context (multiple areas, active placement,
  Shared=Yes) subtract from the score so live reports aren't over-flagged.
- **Recurrence** is inferred from execution **timestamps only** (burst-collapsed,
  cadence-matched: daily/weekly/biweekly/monthly).
- **Duplicate evidence never moves the score.** A configurable weighted model
  (field Jaccard, containment, business object / prompt / related-BO overlap,
  name, data source, authorized usage, report type — renormalized across the
  components that have data) yields a similarity %, relationship label, and a
  recommended keeper.

Recommendation tiers (configurable): 100 → Automatic Decommissioning
Recommendation; 80–99 → High-Priority Review; 60–79 → Decommissioning Review;
40–59 → Owner Review / Monitor; 0–39 → Keep. Duplicate flags adjust the
recommendation **wording**, not the number.

## Configuration

All tunables live in `config.yaml` and are validated on startup
(`config.validate_config` — fails clearly on bad duplicate-weight sums, missing
sections, overlapping recommendation thresholds, or invalid caps). Key sections:
`scoring`, `overall`, `business_protection`, `recurrence`, `recommendation`,
`duplicate_weights`, `duplicate_thresholds`, `hard_rules`, `dedup`, `aliases`.

## Tests

```bash
pytest -q
```

## Security & deployment notes

This tool handles sensitive HR metadata. Key hardening (see `security.py`):

- **Authentication** is off by default for local/offline use. Set
  `REPORT_CLEANUP_REQUIRE_AUTH=1` to gate the dashboard. Configure Streamlit
  OIDC (`[auth]` in `.streamlit/secrets.toml`, e.g. Microsoft Entra ID) for
  per-user SSO, or set `REPORT_CLEANUP_ACCESS_CODE` for a shared-code fallback.
- **Sensitive mode** (`--sensitive` / `REPORT_CLEANUP_SENSITIVE_MODE=1`) masks
  person-identifying columns in the preview, Excel output, **and** the SQLite DB,
  and pseudonymizes the owner filter. Free-text columns can still embed
  identifiers, so output dirs are created `0o700`. The DB is not encrypted —
  store outputs on an encrypted volume for stronger at-rest protection.
- **Uploads** are validated before parsing: byte-size cap, magic-byte signature
  check (extension is not trusted), and an `.xlsx` decompression-bomb guard
  (`security.validate_upload`). The Streamlit server upload cap lives in
  `.streamlit/config.toml`.
- **Error details** are suppressed in the UI (`client.showErrorDetails = "none"`).

**Packaging for distribution:** never hand-zip the working directory (it carries
`.git/`, `.venv/`, caches, `.DS_Store`). Produce a clean archive with:

```bash
git archive --format=zip -o report-decommissioning.zip HEAD
```

`.gitattributes` (`export-ignore`) further strips tests/scripts/dev metadata.

## Module map

| File | Responsibility |
|---|---|
| `config.yaml` / `config.py` | All tunables + startup validation |
| `schema.py` | Canonical fields + header auto-mapping |
| `io_readers.py` | Read csv/xls/xlsx |
| `validate.py` | Required-field checks + duplicate-name detection |
| `clean.py` | Type coercion; exact `normalize_report_name`; helpers |
| `exec_rollup.py` | Runs → per-report rollup (counts, dates, requesters, timestamps) |
| `join.py` | Exact-name join + effective last-run date |
| `recurrence.py` | Timestamp-only recurring-execution detection |
| `hard_rules.py` | DNU (word-boundary) + orphaned worklet → score 100 |
| `soft_scoring.py` | Cleanup-risk rubric (0–60) + reasons |
| `protection.py` | Business-protection credits |
| `overall.py` | Overall Decommissioning Score assembly |
| `field_rollup.py` | Fields/Where_Used → per-report field sets (row-based IDs) |
| `dedup.py` | Inverted-index candidates + union-find duplicate groups |
| `duplicate_similarity.py` | Weighted similarity + relationship labels |
| `recommend.py` | Keeper selection + recommendation wording |
| `flags.py` | Ownership / data-quality flags (no score weight) |
| `db.py` | SQLite persistence |
| `export_excel.py` | Multi-tab workbook |
| `pipeline.py` / `run.py` | Orchestration + CLI |
| `app.py` | Streamlit dashboard |
| `security.py` | Size caps, dir hardening, Excel-injection guard, masking |
