# Seller Copilot (Stage 1: Data Acquisition and Curation)

This repository is now reset to a fresh Stage 1 implementation focused on clean Amazon Home & Kitchen data acquisition, strict quality gates, and curated data outputs for downstream agents/models.

## Locked data source
- Dataset: `McAuley-Lab/Amazon-Reviews-2023`
- Category: `Home_and_Kitchen`
- Paths:
  - `raw/review_categories/Home_and_Kitchen.jsonl`
  - `raw/meta_categories/meta_Home_and_Kitchen.jsonl`

**Full raw files (~43 GB total)** — download once to disk, then run two-pass curation locally:

```bash
python seller-copilot/scripts/download_amazon_hk_raw.py
```

Writes to `seller-copilot/data/raw/amazon_reviews_2023/` (gitignored). Ensure ~50 GB free space.

### After download: build the agent / model dataset (local, no cloud)

**Recommended default:** `config/stage1_local_agent.yaml` — same filters and curation, writes **only Parquet** (no GCP). Use this for demos, models, and the multi-agent app.

```bash
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_local_agent.yaml
```

**What you get** (under `seller-copilot/data/agent_dataset/`, also duplicated under `artifacts/stage1/`):

| File | Use |
|------|-----|
| `products.parquet` | Product metadata + stats for agents / tabular models |
| `reviews.parquet` | Review text for NLP / sentiment / fine-tuning |
| `product_signals.parquet` | Per-product aggregates (counts, sentiment, recency) |
| `retrieval_corpus.parquet` | RAG / embeddings (`product_document` + ids) |
| `agent_dataset_manifest.json` | Row counts, gate settings, column names |

Details: `docs/AGENT_DATASET.md` and `data/agent_dataset/README.md`.

Other configs:

- `config/stage1_amazon_hk.yaml` — HTTP partial ingest (CI).
- `config/stage1_amazon_hk_local.yaml` — local JSONL + optional BigQuery (`ops.require_bigquery`).

`ingestion.max_reviews: 0` / `max_meta: 0` means **no cap** on rows collected for selected products (smoke: set `STAGE1_MAX_REVIEWS` / `STAGE1_MAX_META`).

Expect **hours** of CPU/disk time for a full pass over ~43 GB.

### Generate data (agent + ops)

Pools titles/review text from your local **Home & Kitchen** JSONL, then writes **products / reviews / signals / retrieval** plus **inventory / sales / marketing**:

```bash
python seller-copilot/scripts/generate_data.py --products 8000
python seller-copilot/scripts/generate_data.py --quick
```

Tune `--pool-reviews` / `--pool-meta` for larger text pools (slower pool step). Outputs: `data/agent_dataset/` and `data/synthetic/`.

### Ops data only (if you already have `products.parquet`)

```bash
python seller-copilot/scripts/generate_ops_data.py
```

See **`docs/SYNTHETIC_DATA.md`**. Outputs: `seller-copilot/data/synthetic/` (Parquet + manifest).

### External trends + web context (agent design)

How to combine **RAG + your data + live web search** in one system: **`docs/AGENT_ARCHITECTURE_TRENDS.md`** (single “market intelligence” agent is a good pattern).

## New structure
- `src/common/bq.py`: BigQuery utilities
- `src/data_acquisition/config.py`: typed config loader
- `src/data_acquisition/quality.py`: normalization + quality gates
- `src/data_acquisition/local_jsonl.py`: local two-pass JSONL streaming
- `src/data_acquisition/flows/stage1_amazon_hk_flow.py`: Prefect flow
- `src/data_acquisition/scripts/run_stage1.py`: CLI entrypoint
- `config/stage1_amazon_hk.yaml`: default (remote HTTP ingest)
- `config/stage1_local_agent.yaml`: **local files → agent Parquet bundle (no GCP)**
- `config/stage1_amazon_hk_local.yaml`: full local file ingest (optional BigQuery)
- `scripts/generate_data.py`: **agent + ops Parquet**
- `scripts/generate_ops_data.py`: **ops only** (needs `products.parquet`)
- `docs/SUBMISSION.md`, `docs/SYNTHETIC_DATA.md`, `docs/AGENT_ARCHITECTURE_TRENDS.md`

## Quality gates enforced
- Required IDs, non-empty metadata/text
- Minimum review count per product
- Recency threshold on review timestamps
- Price outlier control (percentile bounds)
- De-duplication and text completeness
- Curation by product quality score

## Run Stage 1
Install dependencies:
```bash
pip install -r seller-copilot/requirements.txt
```

Default (HTTP ingest; CI):
```bash
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_amazon_hk.yaml
```

Local agent bundle (after raw download):
```bash
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_local_agent.yaml
```

Full local files + optional BigQuery:
```bash
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_amazon_hk_local.yaml
```

## Outputs
- **Primary handoff for agents/models** (`seller-copilot/data/agent_dataset/`):
  - `products.parquet`, `reviews.parquet`, `product_signals.parquet`, `retrieval_corpus.parquet`, `agent_dataset_manifest.json`
- **Debug / duplicate copies** (`seller-copilot/artifacts/stage1/`):
  - `products_curated.parquet`, `reviews_curated.parquet`, `product_signals_curated.parquet`, `curation_summary.json`
- **BigQuery** (only if `ops.require_bigquery: true` and credentials set): `products`, `reviews`, `product_signals`, `retrieval_corpus`

## BigQuery (optional)
- `ops.reset_bigquery_tables: true` drops and rebuilds Stage 1 tables when loading cloud.
- `ops.require_bigquery: false` (e.g. `stage1_local_agent.yaml`) skips GCP entirely.

**Upload local Parquet to BigQuery** (agent + synthetic tables): `docs/GCP_SETUP.md` and `scripts/upload_to_bigquery.py`.

**Validate datasets:** `python seller-copilot/scripts/validate_agent_data.py`

