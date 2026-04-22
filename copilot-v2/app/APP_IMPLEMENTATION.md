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

## Pipeline, Orchestrator, and API (Implemented)

### `llm.py`
Lightweight self-contained LLM utility — no third-party SDK.
- `OllamaClient` — posts to the Ollama `/api/chat` endpoint
- `OllamaChatResult` — typed result with `content` and `elapsed_s`
- `extract_json_object()` — extracts the first valid JSON object from a raw model response
- Schema validators: `validate_ranked_actions()`, `validate_specialist_proposal()`, `validate_peer_review()`
- `SchemaError` — raised when a parsed object fails schema validation

### `agents/orchestrator/advocate.py`
- `build_messages()` — builds the Ollama chat messages with `is_revision` flag
  - `is_revision=False` (round 1): task is "argue for the strongest plan"
  - `is_revision=True` (revision rounds): surfaces `round1_critic` and optional `human_feedback` prominently as `CRITIC_FEEDBACK:` in the prompt
  - Payload is slimmed to only the fields the LLM needs (product_id, pricing source, sentiment scores, inventory status, signals) and serialized as proper JSON via `json.dumps()` — not Python dict syntax
- `run()` — calls Ollama, retries twice on JSON parse failure, falls back to top-k candidates by retrieval order

**Key implementation note:** The payload must be serialized with `json.dumps()` before embedding in the prompt. Passing a raw Python dict (with single quotes, `True/False`, `None`) caused llama3.1:8b to describe the input structure instead of producing an action plan.

### `agents/orchestrator/critic.py`
- `build_messages()` — receives the advocate proposal and challenges it; payload slimmed to goal, constraints, advocate output, and candidate inventory/availability signals; serialized with `json.dumps()`
- Output schema: `{agreements: [...], disagreements: [...], suggested_changes: [...]}`
- `run()` — calls Ollama, retries once on failure, falls back to empty peer review

### `agents/orchestrator/judge.py`
- `build_messages()` — receives final advocate + critic outputs plus any human feedback
- `_fallback_from_baseline()` — returns the baseline ranked actions if JSON fails
- `run()` — returns `(final_actions, raw_trace, judge_fallback: bool)`

### `agents/orchestrator/human_review.py`
Retained for reference. The UI now owns the round-continue / move-on decision, so this module is not called by the orchestrator at runtime. The `decide()` function and its three modes (`skip`, `second_round`, `second_round_with_feedback`) remain available for offline testing or future CLI use.

### `agents/orchestrator/orchestrator.py`
Three public functions — no judge runs except in `run_judge_only()`:

```
run_acj()
  Round 1: Advocate → Critic
  Returns: (adv_result, crit_result, debate_trace)
  Writes:  4_debate_advocate_r1.json, 5_debate_critic_r1.json

continue_acj(prev_advocate, prev_critic, human_feedback?)
  Revision round: Advocate (sees prev critic as CRITIC_FEEDBACK) → Critic
  Returns: (adv_result, crit_result, raw_trace)
  Writes:  4_debate_advocate.json, 5_debate_critic.json
  (callable N times by the UI via /debate/continue)

run_judge_only(latest_advocate, latest_critic, human_feedback?)
  Judge runs once on final debate outputs
  Returns: (final_actions, judge_raw, judge_fallback)
  Writes:  8_debate_judge.json
```

`_run_round()` is the shared inner helper used by both `run_acj` and `continue_acj`.

### `pipeline.py`
Four public methods matching the four active API endpoints:

**`run_pipeline()`** — called by `POST /pipeline`:
1. `RetrievalAgent.retrieve(goal)` → candidate list; writes `1_retrieval.json`
2. `_enrich()` → attaches pricing, sentiment, inventory signals; writes `2_enriched.json`
3. `_make_baseline()` → sorts by retrieval score, `price_change=0`; writes `3_baseline.json`
4. Returns `enriched_candidates` + `baseline_ranked_actions` — no LLM calls

**`start_debate()`** — called by `POST /debate/start`:
- Calls `run_acj()` with pre-enriched candidates (no retrieval re-run)
- Creates `run_{ts}_{owner}_debate/` folder; writes stages 4–5
- Returns `{ok, advocate, critic, debate_trace}`

**`continue_debate()`** — called by `POST /debate/continue`:
- Calls `continue_acj(prev_advocate, prev_critic, human_feedback?)`
- Creates a separate `run_{ts}_{owner}_cont/` folder
- Returns `{ok, advocate, critic, raw}` — no ranked_actions, no judge

**`run_judge()`** — called by `POST /debate/judge`:
- Calls `run_judge_only(latest_advocate, latest_critic, human_feedback?)`
- Calls `_merge_judge_output()` to attach enriched signals to judge decisions
- Creates a `run_{ts}_{owner}_judge/` folder; writes `8_debate_judge.json` and `9_final.json`
- Returns `{ok, ranked_actions, judge_raw, judge_fallback}`

### Artifact saving
Every pipeline call writes its outputs to a timestamped folder under `artifacts/runs/{snapshot_id}/`. Files are written immediately after each agent completes so partial runs are recoverable on crash.

| Folder suffix | Created by | Files |
|---|---|---|
| `run_{ts}_{owner}/` | `run_pipeline()` | `1_retrieval.json`, `2_enriched.json`, `3_baseline.json` |
| `run_{ts}_{owner}_debate/` | `start_debate()` | `4_debate_advocate_r1.json`, `5_debate_critic_r1.json` |
| `run_{ts}_{owner}_cont/` | `continue_debate()` | `4_debate_advocate.json`, `5_debate_critic.json` |
| `run_{ts}_{owner}_judge/` | `run_judge()` | `8_debate_judge.json`, `9_final.json` |

### `api/app.py`
FastAPI server with CORS enabled.
- `GET /health` — reports cache load status for all three specialist agents
- `POST /pipeline` — fast retrieval + enrichment + baseline (no LLMs)
- `POST /debate/start` — advocate + critic round 1 (LLMs); takes pre-enriched candidates
- `POST /debate/continue` — one more advocate + critic revision round (LLMs)
- `POST /debate/judge` — runs the judge once on the final advocate + critic; returns `ranked_actions`
- `POST /orchestrate` — legacy all-in-one endpoint; kept for compatibility
- Lazy `get_pipeline()` — initialized once on first request using env vars:
  - `COPILOT_SNAPSHOT_ID` (default: `38710839ca6e1009`)
  - `COPILOT_ARTIFACTS_ROOT` (default: `copilot-v2/artifacts`)
  - `COPILOT_OLLAMA_URL` (default: `http://localhost:11434`)

### Starting the API server (Windows PowerShell)
```powershell
cd copilot-v2
$env:COPILOT_ARTIFACTS_ROOT = "$PWD\artifacts"
$env:PYTHONPATH = "$PWD"
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Risk level calculation (UI)
`buildPlansFromRanked` in `App.jsx` computes risk level from three signals:
- `inventory.risk_flag = true` → **High**
- `sentiment.p_neg > 0.4` → **High**
- `signals.total_returns > 3` or `sentiment.p_neg > 0.2` → **Medium**
- Otherwise → **Low**

---

## What Is Missing / Incomplete

### `recommended_price_change_pct` — not missing, it is a computed label
This column is **generated on demand** by `build_pricing_training_table.py` using a deterministic formula (elasticity-based grid search over ±10% price change candidates). It was never a raw data column — it does not need to be "recovered" from anywhere.

To regenerate the pricing training table:
```bash
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_pricing_training_table \
  --snapshot-id 38710839ca6e1009 \
  --artifacts-root copilot-v2/artifacts
```

This writes `artifacts/evals/{snapshot_id}/pricing/pricing_training_table.parquet`, which `build_pricing_cache.py` then reads.

### Inventory cache does not exist yet
`artifacts/caches/{snapshot_id}/inventory/` does not exist. Unlike pricing and sentiment, the inventory cache was never precomputed. Run `build_inventory_cache.py` once to generate it before `inventory_agent.py` can be used.

### Retrieval agent index persistence *(resolved)*
`RetrievalAgent` now loads the pre-built FAISS index from disk via `load_index()`. The index files already exist at `artifacts/indexes/{snapshot_id}/dense/intfloat_e5-large-v2/`. `build_index()` is only needed when the catalog or model changes.

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
