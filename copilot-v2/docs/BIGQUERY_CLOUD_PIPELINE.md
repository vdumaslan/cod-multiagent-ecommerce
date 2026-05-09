# BigQuery Cloud Data Pipeline

This is the cloud data pipeline for the current `copilot-v2` implementation. It does not use the old `seller-copilot` architecture as truth.

## Target Flow

```text
Local/ETL artifacts for snapshot 38710839ca6e1009
  -> BigQuery structured tables
  -> validation gates
  -> FastAPI startup hydrates caches from BigQuery
  -> runtime serves in-memory lookups
  -> seller chooses a plan in the UI
  -> POST /runs/log writes final decision back to BigQuery operator_decision_log
```

The runtime still uses offline feature serving. Pricing, sentiment, and inventory outputs are precomputed once, stored as tables, and loaded into dictionaries at backend startup. User requests do not query BigQuery for every product.

Local files (`artifacts/runs/`) are always written regardless of BigQuery mode — the cloud write is additive.

## BigQuery Tables

Every uploaded table includes:

- `snapshot_id`
- `ingested_at`
- original Parquet columns

Serving tables:

- `product_metadata`
- `product_signals`
- `product_features`
- `retrieval_corpus`
- `pricing_cache`
- `sentiment_cache`
- `inventory_cache`
- `inventory_skus`

Full warehouse tables:

- all serving tables
- `reviews`
- `products_split`
- `reviews_split`
- `sales_daily`

Control/audit tables:

- `pipeline_runs` — one row per upload run, tracks status and tables loaded
- `data_quality_results` — per-table validation check results written during upload
- `operator_decision_log` — one row per ranked action when a seller chooses a plan in the UI; written by `POST /runs/log` at decision time

## What Stays Local

- FAISS index stays local — runtime query encoding cannot be precomputed.
- `intfloat/e5-large-v2` embedding model stays local for the same reason.
- Ollama debate models stay local unless a paid cloud GPU VM is explicitly provisioned.
- `artifacts/runs/` output files are always written locally regardless of `COPILOT_DATA_BACKEND`.

This avoids accidental GCS or GPU costs.

## Environment

Required for cloud upload/runtime:

```powershell
$env:GCP_PROJECT_ID = "your-gcp-project"
$env:BIGQUERY_DATASET = "copilot_v2"
$env:BIGQUERY_LOCATION = "US"
$env:COPILOT_SNAPSHOT_ID = "38710839ca6e1009"
$env:PYTHONPATH = "copilot-v2/src"
```

Authentication can use either:

- `gcloud auth application-default login` locally
- `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON
- GitHub secret `GCP_SA_KEY_JSON` in the workflow

## Dry Run

Use dry-run first. It does not contact GCP.

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv-copilot-v2\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set serving `
  --dry-run
```

## Local Validation

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv-copilot-v2\Scripts\python.exe -m copilot_v2.scripts.cloud.validate_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set serving `
  --local-only
```

Expected core checks:

- `product_metadata`: 50,000 rows
- `product_features`: 50,000 rows
- `retrieval_corpus`: 50,000 rows
- `pricing_cache`: 35,259 rows
- `sentiment_cache`: 50,000 rows
- `inventory_cache`: 50,000 rows
- no duplicate product IDs in product-level tables
- cache product IDs are subsets of `product_metadata`

## Upload

Free-safe serving upload:

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv-copilot-v2\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set serving `
  --no-dry-run `
  --write-validation-results
```

Full warehouse upload:

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv-copilot-v2\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set full `
  --no-dry-run `
  --write-validation-results
```

Use `full` only if the BigQuery sandbox storage limit can handle the reviews and sales tables.

## Runtime BigQuery Mode

Local fallback is enabled by default. Use the actual GCP project ID.

```powershell
$env:COPILOT_DATA_BACKEND = "bigquery"
$env:COPILOT_BIGQUERY_FALLBACK_TO_LOCAL = "1"
$env:GCP_PROJECT_ID = "linear-theater-436300-r9"
$env:BIGQUERY_DATASET = "copilot_v2"
$env:BIGQUERY_LOCATION = "US"
```

Start the API the same way as before. At startup:

- pricing cache loads from `pricing_cache`
- sentiment cache loads from `sentiment_cache`
- inventory cache loads from `inventory_cache`
- if BigQuery fails, local Parquet is used when fallback is enabled

To force cloud-only behavior:

```powershell
$env:COPILOT_BIGQUERY_FALLBACK_TO_LOCAL = "0"
```

The UI top-right shows a **data backend badge** alongside the API status badge:

- **☁ BigQuery** (blue) — all three caches loaded from BigQuery
- **☁ mixed** (orange) — BigQuery attempted but at least one cache fell back to local
- **⬡ local** (grey) — all caches loaded from local Parquet files

### Decision Logging

When the seller chooses a plan and rates their confidence in the UI, the frontend calls `POST /runs/log`. In BigQuery mode this appends one row per ranked action to `operator_decision_log` with:

| Field | Value |
|---|---|
| `run_id` | pipeline run identifier |
| `snapshot_id` | `38710839ca6e1009` |
| `owner_id` | seller identifier |
| `goal` | the original query |
| `variant` | A/B variant assigned |
| `event` | `decision_submitted` |
| `product_id` | per ranked action |
| `accepted` | `true` for the chosen plan, `false` for others |
| `confidence_rating` | 1–5 from the UI slider |
| `metadata_json` | `action_type`, `title`, `rank` (price pct omitted pending field rename in Bug 2) |

In local mode `POST /runs/log` returns `{"ok": true, "skipped": true}` — no BigQuery write occurs and no error is raised.

## GitHub Actions

Manual workflow:

```text
.github/workflows/copilot-v2-bigquery-pipeline.yml
```

Required secrets for real upload:

- `GCP_PROJECT_ID`
- `GCP_SA_KEY_JSON`
- optional `BIGQUERY_DATASET`
- optional `BIGQUERY_LOCATION`

The workflow defaults to dry-run and `serving` scope. Keep it manual unless you are sure the artifacts are available to the runner and quota usage is acceptable.

## Presentation Summary

The final architecture is:

```text
ETL/artifact snapshot
  -> BigQuery warehouse tables (upload pipeline)
  -> BigQuery validation gates (data quality checks)
  -> backend startup hydrates specialist caches from BigQuery
  -> runtime uses FAISS + in-memory specialist signals
  -> Advocate/Critic/Judge generate ranked plans
  -> seller chooses a plan + rates confidence in UI
  -> POST /runs/log writes decision to BigQuery operator_decision_log
  -> artifacts/runs/ always written locally in parallel
```

This is production-style because it separates offline computation from online serving, keeps runtime fast, tracks snapshot versions, has data quality checks before the application consumes new data, and closes the loop by persisting seller decisions back to the warehouse.

