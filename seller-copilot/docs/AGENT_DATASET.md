# Agent dataset (local export)

## Purpose

The pipeline applies the **quality gates** in `config/stage1_local_agent.yaml` (and env overrides `STAGE1_*`) to the full Home & Kitchen vertical, then writes a **manageable subset** you can use without BigQuery or cloud:

- **Tabular / gradient boosting:** join `products.parquet` with `product_signals.parquet` on `product_id`.
- **Text / NLP:** `reviews.parquet` (and optionally `products.parquet` for product-side text).
- **RAG / embeddings:** `retrieval_corpus.parquet` (`product_document` is title + description).

## Output paths (canonical)

All paths are relative to the repo root:

| Path | Description |
|------|-------------|
| `seller-copilot/data/agent_dataset/products.parquet` | Curated products |
| `seller-copilot/data/agent_dataset/reviews.parquet` | Curated reviews |
| `seller-copilot/data/agent_dataset/product_signals.parquet` | Per-product signals |
| `seller-copilot/data/agent_dataset/retrieval_corpus.parquet` | RAG text index |
| `seller-copilot/data/agent_dataset/agent_dataset_manifest.json` | Metadata + column names |

## Command

```bash
python seller-copilot/scripts/generate_data.py --quick
```

Or full Stage 1 scan over raw JSONL:

```bash
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_local_agent.yaml
```

## Gates (defaults)

Documented in the YAML and echoed in `agent_dataset_manifest.json`:

- Minimum reviews per product, title/review length, rating range, recency year floor.
- Price winsorization (percentile band).
- Cap on number of products (`max_products`) ranked by a quality score after filtering.

Tuning for smaller runs: `STAGE1_MAX_PRODUCTS=5000` etc.

## Synthetic operations data (not from Amazon)

After this bundle exists, generate **inventory / COGS / sales / marketing** tables joined on `product_id`:

```bash
python seller-copilot/scripts/generate_ops_data.py
```

See `docs/SYNTHETIC_DATA.md`.

## Cloud later

The same curated frames are still written to `artifacts/stage1/`; enabling `require_bigquery: true` in a separate config only adds BigQuery load and does not change the local agent bundle.
