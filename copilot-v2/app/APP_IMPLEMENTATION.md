# App Implementation Notes

## Overview

This document covers what was implemented in the `app/` directory, the model and data source decisions made, and what is still missing or incomplete.

---

## Model Selections (from `docs/MODEL_TRAINING_AND_RESULTS.md`)

| Agent | Model | Key Config |
|---|---|---|
| Retrieval | `intfloat/e5-large-v2` | `max_seq_length=384`, `use_prefixes=True` |
| Pricing | TabPFN | `n_estimators=4`, `softmax_temperature=0.85`, `max_fit_rows=21119`, `policy_bound=±10%` |
| Sentiment | `distilroberta-base` (fine-tuned, 500k) | checkpoint: `distilroberta-base_final_500k_slice_winner` |
| Inventory | Rule-based (no model) | 4-class: `stockout_risk`, `low_stock`, `overstocked`, `healthy` |
| Advocate | `llama3.1:8b` | via Ollama |
| Critic | `qwen2.5:7b-instruct` | via Ollama |
| Judge | `qwen2.5:7b-instruct` | via Ollama |
| Prompt style | `few_shot_json` + `v1` | set via `COPILOT_V2_ACJ_PROMPT_STYLE` / `COPILOT_V2_ACJ_PROMPT_VERSION` |

---

## Data Sources

Each agent uses a purpose-built parquet file to avoid cross-pipeline contamination and ensure full product coverage.

| Agent | Source file(s) |
|---|---|
| Retrieval | `artifacts/data_snapshots/{snapshot_id}/retrieval_corpus.parquet` |
| Sentiment precompute | `artifacts/data_snapshots/{snapshot_id}/reviews.parquet` |
| Pricing precompute | `artifacts/features/{snapshot_id}/tabular_features.parquet` |
| Inventory precompute | `artifacts/synthetic/{snapshot_id}/inventory_skus.parquet` + `sales_daily.parquet` |

**Why `tabular_features.parquet` for pricing:** This is almost certainly the same table TabPFN was trained on — it has a `split` column, the pricing manifest references a training table derived from it, and the feature columns match. Using it for inference guarantees the feature schema is identical to what the model saw during training. Coverage is limited to ~50k products as a known trade-off.

**Files not used at runtime:**
- `priced_subset/` variants — training/eval subsets only
- `reviews_sentiment_balanced.parquet` — balanced training split used during model training only
- `product_signals.parquet`, `products.parquet` — covered by `tabular_features.parquet` for pricing; not needed separately

---

## What Was Implemented

### `agents/retrieval_agent.py`
- `RetrievalConfig` dataclass holding the winning model config (`e5-large-v2`, `max_seq_length=384`, `use_prefixes=True`, `top_k=10`)
- `RetrievalAgent` class with three methods:
  - `load_index()` — loads the pre-built FAISS index from `artifacts/indexes/{snapshot_id}/dense/intfloat_e5-large-v2/` (fast, seconds). Returns `True` if successful. **This is the normal startup path.**
  - `build_index(corpus, save=True)` — encodes the full corpus and builds a new FAISS index. Saves to disk automatically. Only needed when the product catalog or model changes.
  - `retrieve(query)` — encodes the user query (fast, single text) and searches the index for top-k results
- The pre-built index already exists in the repo (`index_flatip.faiss`, `corpus.parquet`, `doc_emb_norm_f32.npy`, `index_meta.json`), so `load_index()` succeeds immediately on first startup with no encoding required
- Applies `"query: "` / `"passage: "` prefixes at encode time per the e5 model spec

### `precompute/precompute_pricing.py`
- Loads `tabular_features.parquet` and the saved TabPFN model from `artifacts/models/{snapshot_id}/pricing/tabpfn/model.tabpfn_fit.zip`
- Runs inference on all ~50k rows, clips predictions to `±10%`
- Writes `pricing_cache.json`, `pricing_cache.parquet`, `pricing_cache_manifest.json` to `artifacts/caches/{snapshot_id}/pricing/`

### `precompute/build_pricing_features.py` *(not yet active)*
- Joins `products.parquet` + `product_signals.parquet` + `inventory_skus.parquet` + `sales_daily.parquet` to build a full-coverage feature table
- Kept as a starting point for the team member who owns the feature engineering pipeline to validate and confirm the column transformations match what TabPFN was trained on
- Once confirmed, `precompute_pricing.py` can be switched to use `pricing_features.parquet` for full product coverage

### `precompute/precompute_sentiment.py`
- Loads `reviews.parquet` and the fine-tuned DistilRoBERTa checkpoint from `artifacts/models/{snapshot_id}/sentiment/distilroberta-base_final_500k_slice_winner/`
- Runs batch inference (`batch_size=256`, `max_length=256`) over all review texts
- Aggregates per product: `n_reviews`, `p_pos`, `p_neu`, `p_neg`
- Writes `sentiment_cache.json`, `sentiment_cache.parquet`, `sentiment_cache_manifest.json` to `artifacts/caches/{snapshot_id}/sentiment/`
- `--no-model` flag available to derive labels from star ratings instead (fast fallback for testing)

### `precompute/precompute_inventory.py`
- Loads `inventory_skus.parquet` (stock levels) and `sales_daily.parquet` (daily revenue/returns)
- Aggregates sales: `mean_daily_revenue` (mean of `gross_revenue_usd`), `total_returns` (sum of `return_units`)
- Joins on `product_id` and applies the same 4-class rule thresholds as `src/copilot_v2/runtime/inventory_agent.py`
- Writes `inventory_cache.json`, `inventory_cache.parquet`, `inventory_cache_manifest.json` to `artifacts/caches/{snapshot_id}/inventory/`

### `agents/pricing_agent.py`
- Loads `pricing_cache.parquet` once at startup into a `dict[product_id → predicted_price_change_pct]`
- Exposes `lookup(product_id)` and `lookup_many(product_ids)` — pure dictionary reads at runtime

### `agents/sentiment_agent.py`
- Loads `sentiment_cache.parquet` once at startup into a `dict[product_id → {n_reviews, p_pos, p_neu, p_neg}]`
- Exposes `lookup(product_id)` and `lookup_many(product_ids)`

### `agents/inventory_agent.py`
- Loads `inventory_cache.parquet` once at startup into a `dict[product_id → {stock_status, risk_flag, ...}]`
- Exposes `lookup(product_id)` and `lookup_many(product_ids)`

---

## Cache Schema

### Pricing cache
```
pricing_cache.parquet    columns: product_id, predicted_price_change_pct
pricing_cache.json       {product_id: float}
pricing_cache_manifest.json  schema_version, snapshot_id, source_table, model_state_path, policy_bound, rows, columns
```

### Sentiment cache
```
sentiment_cache.parquet    columns: product_id, n_reviews, p_pos, p_neu, p_neg
sentiment_cache.json       {product_id: {n_reviews, p_pos, p_neu, p_neg}}
sentiment_cache_manifest.json  schema_version, snapshot_id, source_reviews, approach, label_buckets, rows, columns
```

### Inventory cache
```
inventory_cache.parquet    columns: product_id, stock_status, risk_flag, on_hand_units, safety_stock_units, available_to_sell, mean_daily_revenue, total_returns
inventory_cache.json       {product_id: {stock_status, risk_flag, ...}}
inventory_cache_manifest.json  schema_version, snapshot_id, source_skus, source_sales, approach, thresholds, classes, rows, columns
```

---

## Precompute Scripts (authoritative)

All precompute scripts live in `src/copilot_v2/scripts/precompute/`. This is the single location for all offline cache-building steps — pricing, sentiment, inventory, and retrieval indexes.

| Script | Purpose |
|---|---|
| `precompute/build_pricing_training_table.py` | Generates `recommended_price_change_pct` labels from `tabular_features.parquet` and writes `pricing_training_table.parquet` |
| `precompute/build_pricing_cache.py` | Loads TabPFN model + training table, runs inference, writes `pricing_cache.parquet` to `artifacts/caches/` |
| `precompute/build_sentiment_cache.py` | Builds `sentiment_cache.parquet` using DistilRoBERTa or star-rating fallback |
| `precompute/build_owner_indexes.py` | Builds per-owner FAISS dense indexes for retrieval |
| `precompute/build_inventory_cache.py` | Rule-based inventory classification, writes `inventory_cache.parquet` to `artifacts/caches/` |

Cache output paths:
- `artifacts/caches/{snapshot_id}/pricing/pricing_cache.parquet`
- `artifacts/caches/{snapshot_id}/sentiment/sentiment_cache.parquet`
- `artifacts/caches/{snapshot_id}/inventory/inventory_cache.parquet`

The `app/precompute/` directory has been removed — `src/copilot_v2/scripts/precompute/` is now the single source of truth.

---

## What Is Missing / Incomplete

### `recommended_price_change_pct` — not missing, it is a computed label
This column is **generated on demand** by `build_pricing_training_table.py` using a deterministic formula (elasticity-based grid search over ±10% price change candidates). It was never a raw data column — it does not need to be "recovered" from anywhere.

To regenerate the pricing training table:
```bash
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.build_pricing_training_table \
  --snapshot-id 38710839ca6e1009 \
  --artifacts-root copilot-v2/artifacts
```

This writes `artifacts/evals/{snapshot_id}/pricing/pricing_training_table.parquet`, which `build_pricing_cache.py` then reads.

### Inventory cache does not exist yet
`artifacts/caches/{snapshot_id}/inventory/` does not exist. Unlike pricing and sentiment, the inventory cache was never precomputed. `precompute_inventory.py` must be run once to generate it before `inventory_agent.py` can be used.

### Retrieval agent index persistence *(resolved)*
`RetrievalAgent` now loads the pre-built FAISS index from disk via `load_index()`. The index files already exist at `artifacts/indexes/{snapshot_id}/dense/intfloat_e5-large-v2/`. `build_index()` is only needed when the catalog or model changes.

### Agents not wired to the orchestrator
The three cache-reading agents (`pricing_agent.py`, `sentiment_agent.py`, `inventory_agent.py`) and the retrieval agent are standalone classes. They are not yet connected to `pipeline.py` or the debate orchestrator in `src/copilot_v2/runtime/`.

---

## Running the Precompute Pipeline

All scripts are run from the repo root with `PYTHONPATH=copilot-v2/src`. Run in this order:

```bash
# Step 1: Generate pricing labels (required before building pricing cache)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_pricing_training_table \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts

# Step 2: Build pricing cache (requires TabPFN model artifact + training table from step 1)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_pricing_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 3: Build sentiment cache (DistilRoBERTa model)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_sentiment_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach distilroberta --write-json
# Fast fallback (star ratings, no model needed):
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_sentiment_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach ratings

# Step 4: Build inventory cache (rule-based, no model needed — can run at any time)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_inventory_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 5: Build per-owner retrieval indexes (requires sentence-transformers + faiss)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_owner_indexes \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --device cpu
```
