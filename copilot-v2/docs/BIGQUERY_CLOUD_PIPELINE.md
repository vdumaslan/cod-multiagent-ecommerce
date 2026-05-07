# BigQuery Cloud Data Pipeline

This is the cloud data pipeline for the current `copilot-v2` implementation. It does not use the old `seller-copilot` architecture as truth.

## Target Flow

```text
Local/ETL artifacts for snapshot 38710839ca6e1009
  -> BigQuery structured tables
  -> validation gates
  -> FastAPI startup hydrates caches from BigQuery
  -> runtime serves in-memory lookups
```

The runtime still uses offline feature serving. Pricing, sentiment, and inventory outputs are precomputed once, stored as tables, and loaded into dictionaries at backend startup. User requests do not query BigQuery for every product.

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

- `pipeline_runs`
- `data_quality_results`
- `operator_decision_log`

## What Stays Local

- FAISS index stays local for the free/sandbox implementation.
- `intfloat/e5-large-v2` stays local because runtime query encoding cannot be precomputed.
- Ollama debate models stay local unless a paid cloud GPU VM is explicitly provisioned.

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
.venv312\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set serving `
  --dry-run
```

## Local Validation

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv312\Scripts\python.exe -m copilot_v2.scripts.cloud.validate_bigquery_snapshot `
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
.venv312\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set serving `
  --no-dry-run `
  --write-validation-results
```

Full warehouse upload:

```powershell
$env:PYTHONPATH = "copilot-v2/src"
.venv312\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set full `
  --no-dry-run `
  --write-validation-results
```

Use `full` only if the BigQuery sandbox storage limit can handle the reviews and sales tables.

## Runtime BigQuery Mode

Local fallback is enabled by default.

```powershell
$env:COPILOT_DATA_BACKEND = "bigquery"
$env:COPILOT_BIGQUERY_FALLBACK_TO_LOCAL = "1"
$env:GCP_PROJECT_ID = "your-gcp-project"
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
  -> BigQuery warehouse tables
  -> BigQuery validation gates
  -> backend startup hydrates specialist caches
  -> runtime uses FAISS + in-memory specialist signals
  -> Advocate/Critic/Judge generate ranked plans
  -> seller decision can be batch-loaded into BigQuery later
```

This is production-style because it separates offline computation from online serving, keeps runtime fast, tracks snapshot versions, and has data quality checks before the application consumes new data.

