# Report Decommissioning & Consolidation System

Local, offline engine that ingests three Workday exports (Comprehensive metadata,
Runs execution history, Fields/Where_Used), joins them on an exact report-name
key, and produces a single **Overall Decommissioning Score (0–100)** per report
with a complete, reproducible reason trail. It also detects recurring execution,
finds duplicate relationships with a deterministic weighted baseline or an
optional local PyTorch classifier, and emits a staff-reviewable Excel workbook,
a SQLite snapshot, and an optional Streamlit dashboard. **Nothing is ever
auto-deleted** — every output is a recommendation.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes CPU-compatible PyTorch (`torch>=2.6,<3`) because an
enabled production classifier needs it at inference time. No torchvision,
torchaudio, transformer, hosted-model, or scikit-learn dependency is used.

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
- **Duplicate evidence never moves the score.** The deterministic weighted
  baseline and the optional PyTorch classifier are supporting evidence only.
  They can affect duplicate flags and recommendation wording, but cannot change
  the numeric Overall Decommissioning Score. `hard_rules.py`, `soft_scoring.py`,
  business-protection credits, and `overall.py` remain deterministic.

Recommendation tiers (configurable): 100 → Automatic Decommissioning
Recommendation; 80–99 → High-Priority Review; 60–79 → Decommissioning Review;
40–59 → Owner Review / Monitor; 0–39 → Keep. Duplicate flags adjust the
recommendation **wording**, not the number.

## Runtime architecture

```text
Workday exports
  -> deterministic cleaning, execution join, recurrence, and field attachment
  -> deterministic hard rules, cleanup risk, business protection, Overall Score
  -> deterministic blocked duplicate-candidate generation
       |-> existing deterministic field/name grouping and keeper selection
       `-> canonical pairwise feature extraction
           -> PyTorch duplicate probability, when enabled with a valid artifact
              OR deterministic weighted similarity baseline
           -> duplicate pair verdict/ranking + deterministic relationship label
  -> merge duplicate evidence (without changing Overall Score)
  -> human-reviewable recommendation
```

The two branches are intentionally separate. Duplicate detection does not feed
back into the score calculation. Existing union-find duplicate groups and keeper
analysis also remain deterministic; the classifier answers the narrower binary
question, "is this candidate pair a duplicate/consolidation candidate?" It does
not replace hard rules, learn the Overall Score, or predict the human-readable
relationship class.

Candidate generation remains the existing efficient, deterministic blocking
stage. It uses normalized-name prefixes and shared field/business-object indexes,
and skips non-discriminative oversized blocks. The deterministic grouping branch
retains the existing cheap gates and strong-name guards, including protection
against mismatched year/ID tokens. The neural network therefore scores only the
blocked candidate set, never every possible report pair.

### Canonical duplicate features

The weighted baseline and PyTorch path share one feature implementation in
`report_cleanup/ml/features.py`:

| Signal | Representation before model normalization |
|---|---|
| Field Jaccard similarity | Set overlap, 0–100%, or unavailable |
| Smaller-report containment | Shared fields / smaller field set, 0–100%, or unavailable |
| Business-object overlap | Jaccard, 0–100%, or unavailable |
| Normalized report-name similarity | De-noised RapidFuzz similarity, 0–100%, or unavailable; deterministic guards remain separate |
| Built-in-prompt similarity | Jaccard, 0–100%, or unavailable |
| Related-business-object overlap | Jaccard, 0–100%, or unavailable |
| Data-source match | 0 or 100%, or unavailable |
| Authorized-usage similarity | Jaccard, 0–100%, or unavailable |
| Report-type match | 0 or 100%, or unavailable |

The deterministic baseline drops unavailable components and renormalizes the
remaining configured weights; it does not turn missing evidence into disagreement.
For model input, each signal is scaled to 0.0–1.0 and immediately followed by a
missingness indicator. A genuine zero is encoded as `(0.0, 0.0)`; unavailable
data is `(0.0, 1.0)`. The result is an ordered 18-dimensional vector. The exact
feature names, order, normalization rule, and schema version are persisted with
the model, and inference refuses to load an incompatible schema.

The compact model is `18 -> 32 -> 16 -> 1` with ReLU activations and a raw logit
output. Training uses `BCEWithLogitsLoss`, positive-class weighting, AdamW,
mini-batches, deterministic seeds where practical, validation-based checkpoint
selection, and early stopping. Sigmoid is applied only when converting logits to
probabilities. Runtime loads one model per pipeline run and scores candidates in
batches under inference mode on CPU by default. Training automatically uses CUDA
when available, but neither training nor inference requires it.

## Configuration

All tunables live in `config.yaml` and are validated on startup
(`config.validate_config` — fails clearly on bad duplicate-weight sums, missing
sections, overlapping recommendation thresholds, or invalid caps). Key sections:
`scoring`, `overall`, `business_protection`, `recurrence`, `recommendation`,
`duplicate_weights`, `duplicate_thresholds`, `ml_duplicate`, `hard_rules`,
`dedup`, `aliases`.

PyTorch duplicate inference is opt-in:

```yaml
ml_duplicate:
  enabled: false
  model_path: "models/duplicate_model.pt"
  decision_threshold: null
  fallback_to_weighted_similarity: true
  batch_size: 512
```

- `enabled: false` uses the existing deterministic weighted implementation.
- With `enabled: true`, a valid artifact is loaded once and used for batched
  candidate inference.
- `decision_threshold: null` uses the validation-selected threshold stored in
  the artifact. A configured probability from 0.0 through 1.0 overrides it; use
  an override only after evaluating that operating point.
- If the artifact is missing, corrupt, or schema-incompatible and fallback is
  `true`, the run emits a clear warning and uses the weighted baseline. With
  fallback `false`, the run fails rather than silently claiming ML ran.

Relative model paths are resolved beside the active config file (the project root
for the bundled `config.yaml`). A clean clone contains no trained model, so the
default disabled configuration is fully deterministic.

Every output record exposes `duplicate_scoring_mode` and
`duplicate_model_status`. Status is `disabled`, `pytorch`, or
`fallback_weighted`; scoring mode is `ml` or `weighted_baseline`. When ML runs,
the record also exposes native 0.0–1.0 `duplicate_ml_probability` and
`duplicate_ml_threshold`, `duplicate_ml_prediction`, `duplicate_model_version`,
and the exact ordered 18-value `duplicate_feature_values` map used for its best
candidate. Excel renders probabilities as percentages and the feature map as the
`ML Feature Evidence` column, preserving `n/a` for missing signals. Existing
field Jaccard, containment, weighted similarity, relationship, and reason-trail
fields remain available.

The pipeline result and field diagnostics include a `duplicate_scoring` block
with status, candidate/qualified pair counts, number scored by ML, model version,
threshold, batch count, and inference time where applicable. These fields prove
which path ran; if loading fails under fallback, the terminal/run warnings say so
as well. Feature values are review context, not causal explanations of the
network.

## Duplicate-model training and evaluation

### Ground truth and labeling

No trustworthy human-reviewed duplicate labels were found in this repository,
and no production model is committed. The existing weighted formula is a
baseline, **not ground truth**: it is never used to fabricate labels or to claim
model quality. The pair-generation command exports only deterministically blocked
candidate pairs with report identifiers, reviewer context, the nine raw signals,
explicit `_missing` columns, the exact normalized 18-value feature vector, the
weighted baseline evidence, and an empty `label` column. Unavailable raw signals
remain blank. The loader recomputes the vector from those raw signals and rejects
schema, order, range, missingness, or cross-column inconsistencies.

A reviewer assigns:

- `1` — a trusted duplicate/consolidation candidate.
- `0` — not a duplicate.
- Blank or `unsure` — unresolved; it must not silently become a negative example.

Keep report identifiers stable while labeling. Training examples are traceable by
both report IDs, report names, source-export SHA-256 values, generation timestamp,
and optional reviewer/notes metadata. Generated labeling CSVs can contain internal
report metadata and are ignored by Git under `data/training/`.

### End-to-end commands

1. Export the same deterministically blocked pairs the runtime can score. The
   fields export is optional to the CLI but strongly recommended; without it,
   field-based signals are unavailable.

   ```bash
   python -m training.generate_pairs \
     --table1 metadata.xlsx \
     --table2 execution.csv \
     --table3-fields fields.xlsx \
     --output data/training/candidate_pairs.csv
   ```

   Use `--limit N` for a deliberately capped first review batch. The command
   reports both the full candidate count and exported count so a cap cannot look
   like complete coverage. It refuses to replace an existing review file unless
   `--overwrite` is explicitly supplied.

2. Open `data/training/candidate_pairs.csv`, review each pair, and fill `label`
   with `1` or `0`. Blank/`unsure` rows are excluded and counted. Any other
   unrecognized label fails with its CSV row number. Do not copy
   `baseline_prediction` into `label`.

3. Train and save the best validation checkpoint:

   ```bash
   python -m training.train_duplicate_model \
     --data data/training/candidate_pairs.csv \
     --output models/duplicate_model.pt \
     --epochs 200 \
     --batch-size 64 \
     --learning-rate 0.001 \
     --weight-decay 0.0001 \
     --seed 42 \
     --unresolved-labels exclude \
     --patience 20 \
     --min-precision 0.90 \
     --split-manifest models/duplicate_model.pt.split.json \
     --summary-json data/training/training_summary.json
   ```

   The defaults are `32 16` hidden units, 70/15/15 group-aware split ratios,
   zero dropout, and automatic CUDA use when available. Pass `--device cpu` to
   force CPU, or see `python -m training.train_duplicate_model --help` for all
   hyperparameters. Use `--unresolved-labels reject` to require a completely
   resolved review instead of excluding/counting unresolved rows. Training
   deliberately does not inspect the test split.

4. After model and threshold selection are complete, evaluate the frozen artifact
   against the deterministic baseline on the held-out test split:

   ```bash
   python -m training.evaluate_duplicate_model \
     --data data/training/candidate_pairs.csv \
     --model models/duplicate_model.pt \
     --split test \
     --split-manifest models/duplicate_model.pt.split.json \
     --output-json data/training/test_evaluation.json
   ```

5. Only after reviewing that comparison, set `ml_duplicate.enabled: true` in
   `config.yaml`, confirm `model_path`, and run the normal pipeline:

   ```bash
   python -m report_cleanup.run \
     --table1 metadata.xlsx \
     --table2 execution.csv \
     --table3-fields fields.xlsx \
     --config config.yaml \
     --out ./output
   ```

   Verify `Duplicate Scoring Mode = ml` and `Duplicate Model Status = pytorch` in
   the workbook (or the returned `duplicate_scoring` diagnostics). A
   `fallback_weighted` status means the configured artifact was not used.

### Leakage, imbalance, and threshold selection

The training split is group-aware. Report pairs form an undirected graph, and
whole connected components are assigned to train, validation, or test (70/15/15
by default). The scripts validate that no report ID appears across splits; a
naive random pair split is never used. Training writes exact row membership to a
companion split manifest and stores its SHA-256 in the artifact. Evaluation
verifies the manifest, labeled-dataset hash, row identities/order, and feature
schema before scoring the frozen held-out rows; it does not recreate a potentially
different split from a seed alone.

Training reports total/positive/negative counts and positive rate for every
split. `BCEWithLogitsLoss` receives a positive-class weight derived from the
training split so rare duplicates are not ignored. The checkpoint with the best
validation average precision is retained instead of the final epoch.

The production threshold is chosen on validation data by maximizing F1 subject
to a configurable precision floor (90% by default). This prioritizes avoiding
false positives that could send a live report toward consolidation review. If no
threshold satisfies the floor, the artifact records that fact explicitly and the
model must not be treated as production-ready merely because training completed.
The held-out test split is used only after checkpoint and threshold selection.

Evaluation reports precision, recall, F1, ROC-AUC when defined, average precision
(PR-AUC) when defined, confusion matrix, and class counts. It evaluates the
existing weighted heuristic and PyTorch probabilities on the **same held-out
labeled examples**. It does not claim the model is better unless those results
show it. Since this repository currently has no reviewed labels, there is no
valid baseline-vs-PyTorch performance comparison yet.

### Artifact format

`models/duplicate_model.pt` is a local state-dict artifact, not a serialized
`nn.Module`. Its plain metadata records the artifact/model/architecture versions,
exact ordered feature schema and normalization, validation-selected threshold and
criterion, training timestamp, PyTorch version, hyperparameters, random seed,
dataset counts, and validation metrics. The held-out comparison is optionally
written to a separate evaluation JSON. Artifact loading uses `weights_only=True`
and validates the schema and architecture before accepting the weights.

Training also writes `models/duplicate_model.pt.split.json` (or the path supplied
with `--split-manifest`). It records the exact train/validation/test membership,
source-dataset SHA-256, unresolved-label policy, split strategy/ratios/seed, and
feature schema. Keep it beside the artifact: held-out evaluation refuses a
missing, changed, or mismatched manifest.

Model files (`.pt`, `.pth`, checkpoints) and generated training CSVs are ignored
by default; `models/.gitkeep` and `data/training/.gitkeep` preserve the intended
directories. Only promote a separately governed artifact after evaluation on
representative human-reviewed data.

### Current model status

- ML infrastructure: implemented for local training, evaluation, persistence,
  and batched inference.
- Smoke testing: synthetic labels may be used by automated tests to prove code
  paths, but their metrics are not evidence of duplicate-detection quality.
- Production model trained: no.
- Production model validated: no. Representative human-reviewed labels are
  required before enabling ML as a production gate.

### Limitations

- Model quality depends on sufficiently large, representative, consistently
  human-reviewed labels containing both classes. Synthetic or heuristic-derived
  labels cannot establish production quality.
- The classifier can only rank pairs admitted by deterministic blocking. It
  cannot recover a true duplicate pair that candidate generation never emits.
- Missingness indicators distinguish unavailable evidence from zero overlap, but
  they cannot recreate missing field or metadata exports.
- Thresholds and metrics reflect the reviewed population at training time.
  Re-evaluate after material Workday/schema changes or label-distribution drift.
- Probabilities and feature values support review; neither is a causal
  explanation or authorization to delete a report.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Security & deployment notes

This tool handles sensitive HR metadata. Key hardening (see `security.py`):

- **ML is local-only.** Pair exports, labels, model training, evaluation, artifact
  loading, and inference make no hosted-model or external-service calls. Treat
  reviewer CSVs and model artifacts as internal data, restrict filesystem access,
  and do not commit them; Git ignores their generated locations by default.
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
| `report_cleanup/ml/features.py` | Canonical nine signals + ordered 18-value model schema |
| `report_cleanup/ml/model.py` | Compact PyTorch binary classifier |
| `report_cleanup/ml/artifact.py` | State-dict metadata persistence + compatibility checks |
| `report_cleanup/ml/inference.py` | One-time loading, fallback policy, and batched prediction |
| `training/generate_pairs.py` | Reviewer-ready unlabeled candidate-pair export |
| `training/dataset.py` / `split.py` | Strict labels, PyTorch dataset, leakage-safe splitting |
| `training/train_duplicate_model.py` | Weighted-loss training, early stopping, threshold selection |
| `training/evaluate_duplicate_model.py` / `metrics.py` | Held-out baseline comparison + dependency-light metrics |
| `models/` / `data/training/` | Ignored local artifacts and sensitive reviewer datasets |
| `recommend.py` | Keeper selection + recommendation wording |
| `flags.py` | Ownership / data-quality flags (no score weight) |
| `db.py` | SQLite persistence |
| `export_excel.py` | Multi-tab workbook |
| `pipeline.py` / `run.py` | Orchestration + CLI |
| `app.py` | Streamlit dashboard |
| `security.py` | Size caps, dir hardening, Excel-injection guard, masking |
