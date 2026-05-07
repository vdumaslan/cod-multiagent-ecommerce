# Copilot V2 Data Pipeline Documentation

## Purpose

The data pipeline prepares the Amazon Home & Kitchen catalog, builds model-ready tables, uploads the structured data to BigQuery, validates the warehouse, and serves the application from BigQuery-backed caches.

The deployed BigQuery dataset is:

```text
Project: linear-theater-436300-r9
Dataset: copilot_v2
Snapshot: 38710839ca6e1009
Location: US
```

## End-To-End Flow

```text
Amazon Home & Kitchen source data
  -> local snapshot artifacts
  -> feature engineering and splits
  -> specialist model precompute
  -> BigQuery warehouse tables
  -> BigQuery validation checks
  -> FastAPI startup cache hydration
  -> React UI recommendations
```

Runtime is not doing expensive model inference for every request. The specialist outputs are precomputed, stored in BigQuery, loaded into memory at backend startup, and then used as fast lookup tables during recommendation generation.

## Pipeline Stages

### Stage 1: Data Snapshot

Raw product and review data is cleaned and converted into canonical Parquet tables.

Outputs:

| Table | Local artifact | BigQuery table | Rows |
|---|---|---:|---:|
| Products | `artifacts/data_snapshots/38710839ca6e1009/products.parquet` | `product_metadata` | 50,000 |
| Reviews | `artifacts/data_snapshots/38710839ca6e1009/reviews.parquet` | `reviews` | 5,310,178 |
| Product signals | `artifacts/data_snapshots/38710839ca6e1009/product_signals.parquet` | `product_signals` | 50,000 |
| Retrieval corpus | `artifacts/data_snapshots/38710839ca6e1009/retrieval_corpus.parquet` | `retrieval_corpus` | 50,000 |

What this stage does:

- Normalizes product IDs and review IDs.
- Cleans product metadata.
- Filters and structures review records.
- Builds product-level signal summaries.
- Builds product text documents for semantic retrieval.

### Stage 2: Feature Engineering And Splits

The pipeline creates model-ready feature tables and train/validation/test splits.

Outputs:

| Table | Local artifact | BigQuery table | Rows |
|---|---|---:|---:|
| Product features | `artifacts/features/38710839ca6e1009/tabular_features.parquet` | `product_features` | 50,000 |
| Product split | `artifacts/splits/38710839ca6e1009/products_split.parquet` | `products_split` | 50,000 |
| Review split | `artifacts/splits/38710839ca6e1009/reviews_split.parquet` | `reviews_split` | 5,310,178 |

What this stage does:

- Adds review aggregates.
- Adds pricing and rating features.
- Adds synthetic sales/inventory features.
- Creates product and review split tables for model evaluation.

### Stage 3: Synthetic Operations Data

The project includes synthetic inventory and sales operations data so the inventory agent can reason about stock position, sales velocity, and return risk.

Outputs:

| Table | Local artifact | BigQuery table | Rows |
|---|---|---:|---:|
| Inventory SKUs | `artifacts/synthetic/38710839ca6e1009/inventory_skus.parquet` | `inventory_skus` | 50,000 |
| Daily sales | `artifacts/synthetic/38710839ca6e1009/sales_daily.parquet` | `sales_daily` | 4,500,000 |

Used by:

- Inventory classification
- Stock risk detection
- Sales velocity signals
- Return risk signals

### Stage 4: Specialist Precompute

Specialist models/rules run offline and produce serving caches.

Outputs:

| Cache | Method | BigQuery table | Rows |
|---|---|---:|---:|
| Pricing cache | TabPFN pricing model | `pricing_cache` | 35,259 |
| Sentiment cache | DistilRoBERTa sentiment model | `sentiment_cache` | 50,000 |
| Inventory cache | Inventory rules | `inventory_cache` | 50,000 |

Pricing cache has 35,259 rows because only products with valid price data are used for pricing recommendations.

### Stage 5: Retrieval Index

Retrieval uses the `retrieval_corpus` table as the structured text source.

Runtime retrieval components:

| Component | Location | Purpose |
|---|---|---|
| `retrieval_corpus` | BigQuery | Product documents and metadata |
| FAISS index | Local artifact | Fast nearest-neighbor search |
| `intfloat/e5-large-v2` | Local model | Runtime query encoding |

The operator query is not known ahead of time, so the E5 model encodes the query at runtime. The product-side index is prebuilt.

### Stage 6: BigQuery Upload

The upload script reads local snapshot artifacts and writes them into BigQuery.

Script:

```text
copilot-v2/src/copilot_v2/scripts/cloud/upload_bigquery_snapshot.py
```

Main command:

```powershell
$env:PYTHONPATH = "copilot-v2/src"
$env:GOOGLE_APPLICATION_CREDENTIALS = "local-docs/linear-theater-436300-r9-681a656e5a07.json"
$env:GCP_PROJECT_ID = "linear-theater-436300-r9"
$env:BIGQUERY_DATASET = "copilot_v2"
$env:BIGQUERY_LOCATION = "US"

.venv312\Scripts\python.exe -m copilot_v2.scripts.cloud.upload_bigquery_snapshot `
  --artifacts-root copilot-v2/artifacts `
  --snapshot-id 38710839ca6e1009 `
  --table-set full `
  --no-dry-run `
  --write-validation-results
```

Supported table sets:

| Table set | Use |
|---|---|
| `caches` | Upload only `pricing_cache`, `sentiment_cache`, `inventory_cache` |
| `serving` | Upload runtime-serving warehouse tables |
| `full` | Upload complete warehouse tables including reviews, splits, and sales |

### Stage 7: Validation

Validation checks run after upload and through GitHub Actions.

Script:

```text
copilot-v2/src/copilot_v2/scripts/cloud/validate_bigquery_snapshot.py
```

Validation checks:

- BigQuery table exists.
- Row counts match expected snapshot counts.
- Product-level tables have unique `product_id`.
- Review-level tables have unique `review_id`.
- Cache product IDs exist in `product_metadata`.
- Retrieval and feature product IDs align with `product_metadata`.

Validation result storage:

```text
copilot_v2.data_quality_results
```

Pipeline run storage:

```text
copilot_v2.pipeline_runs
```

## BigQuery Warehouse Tables

Current BigQuery tables:

| Table | Rows | Role |
|---|---:|---|
| `product_metadata` | 50,000 | Clean product catalog |
| `reviews` | 5,310,178 | Review fact table |
| `product_signals` | 50,000 | Review/rating/price signal table |
| `product_features` | 50,000 | Model-ready feature table |
| `retrieval_corpus` | 50,000 | Product text documents for retrieval |
| `products_split` | 50,000 | Product train/validation/test split |
| `reviews_split` | 5,310,178 | Review train/validation/test split |
| `pricing_cache` | 35,259 | Pricing agent serving cache |
| `sentiment_cache` | 50,000 | Sentiment agent serving cache |
| `inventory_cache` | 50,000 | Inventory agent serving cache |
| `inventory_skus` | 50,000 | Synthetic inventory table |
| `sales_daily` | 4,500,000 | Synthetic sales/returns table |
| `pipeline_runs` | audit table | Pipeline run log |
| `data_quality_results` | audit table | Validation result log |
| `operator_decision_log` | audit table | Human decision logging table |

All uploaded warehouse tables include:

- `snapshot_id`
- `ingested_at`
- original source columns

## Runtime Data Serving

The FastAPI backend supports two data backends:

| Backend | Setting | Behavior |
|---|---|---|
| Local | `COPILOT_DATA_BACKEND=local` | Loads Parquet caches from `copilot-v2/artifacts` |
| BigQuery | `COPILOT_DATA_BACKEND=bigquery` | Loads cache tables from BigQuery at startup |

BigQuery runtime environment:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "local-docs/linear-theater-436300-r9-681a656e5a07.json"
$env:GCP_PROJECT_ID = "linear-theater-436300-r9"
$env:BIGQUERY_DATASET = "copilot_v2"
$env:BIGQUERY_LOCATION = "US"
$env:COPILOT_DATA_BACKEND = "bigquery"
$env:COPILOT_BIGQUERY_FALLBACK_TO_LOCAL = "1"
```

Runtime cache hydration:

```text
BigQuery pricing_cache   -> PricingAgent in-memory dictionary
BigQuery sentiment_cache -> SentimentAgent in-memory dictionary
BigQuery inventory_cache -> InventoryAgent in-memory dictionary
Local FAISS index        -> RetrievalAgent
Local E5 model           -> query encoder
Local Ollama             -> Advocate/Critic/Judge debate
```

After startup, specialist lookups are in-memory. Runtime recommendations do not issue BigQuery queries per candidate.

## Recommendation Flow

```text
User goal from React UI
  -> FastAPI backend
  -> E5 encodes query
  -> FAISS retrieves product candidates
  -> PricingAgent looks up price recommendation
  -> SentimentAgent looks up sentiment summary
  -> InventoryAgent looks up stock/risk status
  -> Advocate/Critic/Judge debate ranks action plans
  -> UI displays recommendation and evidence
```

Signals used in final recommendations:

- retrieval relevance
- predicted price change
- sentiment probabilities
- review volume
- stock status
- risk flag
- available-to-sell units
- mean daily revenue
- total returns
- policy constraints

## GitHub Actions Automation

Workflow:

```text
.github/workflows/copilot-v2-bigquery-pipeline.yml
```

Runs:

- Daily scheduled BigQuery validation for serving tables.
- Manual validation for `caches`, `serving`, or `full`.
- Manual upload mode when artifacts are available on the runner.

GitHub secrets:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | `linear-theater-436300-r9` |
| `GCP_SA_KEY_JSON` | service account JSON contents |
| `BIGQUERY_DATASET` | `copilot_v2` |
| `BIGQUERY_LOCATION` | `US` |

Manual workflow options:

| Input | Values |
|---|---|
| `operation` | `validate_bigquery`, `upload` |
| `snapshot_id` | `38710839ca6e1009` |
| `table_set` | `caches`, `serving`, `full` |
| `dry_run` | `true`, `false` |

The latest GitHub Actions validation run passed against the live BigQuery dataset.

## Operational Refresh Pattern

When new data is available:

```text
1. Rebuild or refresh local artifact snapshot.
2. Run precompute scripts for pricing, sentiment, and inventory caches.
3. Upload refreshed tables to BigQuery.
4. Run validation.
5. Restart backend or start a new backend process.
6. Backend hydrates latest BigQuery caches.
7. Recommendations reflect updated cache values.
```

Inventory-driven updates:

```text
Updated inventory/sales data
  -> inventory_skus and sales_daily refresh
  -> inventory_cache refresh
  -> BigQuery upload
  -> backend cache hydration
  -> recommendation ranking can change
```

Example:

```text
Product A was healthy yesterday.
New inventory data marks it low_stock today.
Inventory cache updates in BigQuery.
Backend loads the new cache at startup.
The recommendation changes from promote/reprice to hold/restock/investigate depending on policy constraints.
```

## Implementation Files

| File | Purpose |
|---|---|
| `copilot-v2/src/copilot_v2/scripts/cloud/cloud_bigquery.py` | BigQuery table specs, upload helpers, validation helpers |
| `copilot-v2/src/copilot_v2/scripts/cloud/upload_bigquery_snapshot.py` | Upload CLI |
| `copilot-v2/src/copilot_v2/scripts/cloud/validate_bigquery_snapshot.py` | Validation CLI |
| `copilot-v2/app/bigquery_cache.py` | Runtime BigQuery cache loader |
| `copilot-v2/app/agents/pricing_agent.py` | Pricing cache backend selection |
| `copilot-v2/app/agents/sentiment_agent.py` | Sentiment cache backend selection |
| `copilot-v2/app/agents/inventory_agent.py` | Inventory cache backend selection |
| `.github/workflows/copilot-v2-bigquery-pipeline.yml` | GitHub Actions automation |

## Current Status

Completed:

- Full snapshot uploaded to BigQuery.
- BigQuery validation passed.
- GitHub Actions validation passed.
- Runtime BigQuery cache hydration tested.
- Local fallback remains available for demo reliability.

Current deployment mode:

```text
BigQuery: structured warehouse and specialist serving caches
Local artifacts: FAISS index and model binaries
FastAPI: runtime orchestration and cache hydration
React: user interface
Ollama: local debate LLM runtime
```
