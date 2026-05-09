# copilot_v2 Source Overview

This document explains the structure of the `copilot_v2` package, how the directories connect, and the full data-to-serving pipeline.

---

## Full Pipeline

```
data_prep/       →  artifacts/data_snapshots/    (raw + synthetic data)
      ↓
training/        →  artifacts/models/            (trained model artifacts)
      ↓
[ human manually updates artifacts/registry/registry.json ]
      ↓
precompute/      →  artifacts/caches/            (inference caches)
      ↓
eval/            →  artifacts/evals/             (quality metrics)
      ↓
app/api/app.py                                   (serves the live system)
```

---

## scripts/data_prep/

Runs first. Prepares raw and synthetic data into `artifacts/data_snapshots/<snapshot_id>/`.

- `stage1_build.py` — ingests raw data, writes `products.parquet`, `reviews.parquet`, `retrieval_corpus.parquet`
- `generate_synthetic_ops.py` — generates synthetic inventory/sales data (`inventory_skus.parquet`, `sales_daily.parquet`) since real ops data is unavailable
- `enrich_snapshot_derived.py` — adds derived fields like `product_document` (text blob used for retrieval)
- `build_splits.py` — assigns train/val/test splits, writes to `artifacts/splits/`
- `balance_reviews_sentiment.py` — undersamples positive reviews to reduce class imbalance before sentiment training
- `export_priced_subset.py` — filters to only products with a valid price, writes to `data_snapshots/<id>/priced_subset/`

---

## scripts/training/

Trains models and saves artifacts to `artifacts/models/<snapshot_id>/`. After a training run, a human manually updates `registry.json` to point at the winning model.

- `train_sentiment_encoder.py` — fine-tunes distilroberta for sentiment classification, saves to `artifacts/models/.../sentiment/`
- `tune_pricing_tabpfn.py` — trains/tunes TabPFN regressor for price change prediction, saves fit state to `artifacts/models/.../pricing/tabpfn/`
- `tune_pricing_catboost.py` / `tune_pricing_ft_transformer.py` — alternative pricing model candidates
- `tune_retrieval_dense.py` — tunes the dense retrieval model (e5-large-v2)

---

## artifacts/registry/registry.json

The handoff point between training and precompute. Records which model won evaluation and where its artifact lives locally. **Not updated automatically** — a human updates it after deciding which trained model to promote.

Key entries:
- `pricing.model_artifact_path` — path to the winning TabPFN fit state zip
- `pricing.train_report_path` — path to the training report (used to reconstruct feature schema at cache build time)
- `sentiment.model_dir` — path to the winning distilroberta model directory
- `agents.retrieval` — index paths and eval metrics for dense retrieval models

---

## scripts/precompute/

Reads `registry.json` to find model paths, loads models from `artifacts/models/`, runs inference, and writes output caches to `artifacts/caches/<snapshot_id>/`.

- `build_sentiment_cache.py` — two modes:
  - `ratings`: rule-based bucketing (neg ≤2, neu =3, pos ≥4), no model needed
  - `distilroberta`: loads the winner model from `registry["sentiment"]["model_dir"]`, runs inference over `reviews.parquet`
- `build_pricing_cache.py` — loads TabPFN fit state from `registry["pricing"]["model_artifact_path"]`, predicts price change % for all products
- `build_pricing_training_table.py` — pure numerical label generation (grid search over a price-change formula), no model needed
- `build_inventory_cache.py` — rule-based only (percentile thresholds on on_hand, safety stock, revenue, returns), no model needed
- `build_owner_indexes.py` — builds per-owner FAISS dense indexes using `intfloat/e5-large-v2` pulled from HuggingFace Hub

### Model sources summary

| Script | Model source |
|---|---|
| `build_sentiment_cache.py` (distilroberta) | Local path from `registry.json` → `artifacts/models/` |
| `build_pricing_cache.py` | Local path from `registry.json` → `artifacts/models/` |
| `build_owner_indexes.py` | HuggingFace Hub (`intfloat/e5-large-v2`) |
| `build_inventory_cache.py` | No model (rule-based) |
| `build_pricing_training_table.py` | No model (numerical) |

---

## scripts/eval/

Runs after training and precompute. Measures quality of models and the debate pipeline.

- `benchmark_llm_policy.py` — benchmarks LLM debate/policy models end-to-end using the full orchestrator
- `cache_specialist_outputs.py` — runs the orchestrator and saves specialist inputs to JSONL for later replay
- `replay_debate_models.py` — replays cached specialist inputs across different advocate/critic/judge model combinations, scores with the debate rubric
- `smoke_owner_retrieval.py` — quick sanity check that retrieval results belong to the correct owner

---

## scripts/utils/

Operational and maintenance tools.

- `orchestrator_server.py` — starts the old HTTP server (legacy, replaced by `app/api/app.py`)
- `rewrite_index_meta_paths.py` — fixes absolute paths in `index_meta.json` files to relative paths (needed when moving artifacts between machines)
- `eda_artifacts_parquet.py` — inspects and profiles parquet files in `artifacts/` for debugging

---

## copilot_v2/runtime, llm, eval (Legacy)

These modules are the original serving implementation and are **no longer the active system**. They have been replaced by the `app/` directory.

- `runtime/orchestrator.py` — original retrieval + ranking logic
- `runtime/debate.py` — original advocate/critic/judge debate loop
- `runtime/server.py` — original HTTP server (raw `http.server`, replaced by FastAPI in `app/`)
- `runtime/cache_io.py` — loads pricing/sentiment caches, supports GCS URIs
- `runtime/inventory_agent.py` — rule-based inventory classifier
- `llm/ollama_client.py` — original Ollama HTTP client
- `llm/json_schema.py` — LLM JSON parsing and validation
- `eval/debate_rubric.py` — deterministic debate output scorer

**These modules are safe to delete** with one prerequisite: `build_owner_indexes.py` imports `_default_owner_for_product` from `runtime/orchestrator.py`. That function (3 lines) must be inlined directly into `build_owner_indexes.py` before the runtime module is deleted.

---

## Active Serving System: app/

The `app/` directory is the current production system, replacing `copilot_v2/runtime` and `llm`.

- `app/api/app.py` — FastAPI app with full routes (health, orchestrate, debate, pipeline, A/B)
- `app/pipeline.py` — main wiring: retrieval → specialist enrichment → ACJ debate
- `app/llm.py` — self-contained Ollama client and JSON extraction (no `src/` imports)
- `app/agents/retrieval_agent.py` — FAISS dense retrieval
- `app/agents/pricing_agent.py` / `sentiment_agent.py` / `inventory_agent.py` — specialist agents
- `app/agents/orchestrator/` — advocate, critic, and judge as separate modules
- `app/ab.py` — A/B testing framework
- `app/rl.py` — contextual bandit RL for model selection
