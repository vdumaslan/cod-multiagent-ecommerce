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
- `RetrievalAgent` class with `build_index(corpus)` and `retrieve(query)` methods
- Encodes corpus using sentence-transformers, builds an in-memory FAISS index
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

## What Is Missing / Incomplete

### `recommended_price_change_pct` target column
The pricing model (TabPFN) was originally trained against a target column called `recommended_price_change_pct`. This column does not exist in any parquet file in the repository:
- `tabular_features_meta.json` lists it under `target_candidates` but it was never added to `tabular_features.parquet`
- `artifacts/evals/` directory does not exist locally — the original pricing training table (`pricing_training_table.parquet`) was never committed

**Impact:** `precompute_pricing.py` currently relies on the pre-saved `model.tabpfn_fit.zip` for inference. If that model file is lost or needs to be retrained from scratch, there is no target column available to re-fit TabPFN.

**Resolution options:**
1. Recover the original `pricing_training_table.parquet` from the team and add it to `artifacts/evals/`
2. Derive `recommended_price_change_pct` from existing signals (e.g. `positive_ratio`, `price_percentile_in_subcategory`) and add it to `tabular_features.parquet` as a new column

### Inventory cache does not exist yet
`artifacts/caches/{snapshot_id}/inventory/` does not exist. Unlike pricing and sentiment, the inventory cache was never precomputed. `precompute_inventory.py` must be run once to generate it before `inventory_agent.py` can be used.

### Retrieval agent index not wired to a persistent store
`RetrievalAgent.build_index()` builds an in-memory FAISS index at runtime. There is no logic to save or load a pre-built index from disk. For production use, the index should be serialized to `artifacts/indexes/{snapshot_id}/dense/intfloat_e5-large-v2/` (that directory already exists with a `corpus.parquet` inside).

### Agents not wired to the orchestrator
The three cache-reading agents (`pricing_agent.py`, `sentiment_agent.py`, `inventory_agent.py`) and the retrieval agent are standalone classes. They are not yet connected to `pipeline.py` or the debate orchestrator in `src/copilot_v2/runtime/`.

---

## Running the Precompute Pipeline

Run in this order (inventory has no dependency on others):

```bash
# Pricing — uses tabular_features.parquet (~50k products, same schema TabPFN was trained on)
python -m app.precompute.precompute_pricing

# Sentiment — uses fine-tuned DistilRoBERTa checkpoint
python -m app.precompute.precompute_sentiment
# Fast fallback (star ratings instead of model):
python -m app.precompute.precompute_sentiment --no-model

# Inventory — rule-based, no model needed
python -m app.precompute.precompute_inventory
```

All scripts default to `snapshot_id=38710839ca6e1009` and resolve paths relative to the `artifacts/` directory.
