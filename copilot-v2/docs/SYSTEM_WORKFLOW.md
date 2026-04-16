# System workflow (owner-scoped orchestrator) — reproducible guide

This document is the **single source of truth** for how the `copilot-v2` system turns a **business goal** into a **ranked plan of SKU-level actions**, scoped to an `owner_id` (store) and grounded in retrieval + signals.


---

## Big picture

At runtime, the system answers the question:

> “Given a business goal for **store X**, which **products owned by store X** should we act on, and what should we do?”

### Core ideas
- **Owner-scoped retrieval**: we retrieve candidates **within the owner’s catalog** (not global-then-filter).
- **Grounding first**: every recommendation is tied to a `product_id` with evidence and attached signals.
- **Optional LLM policy**: an LLM can refine the plan, but is constrained to only choose from grounded candidates.

---

## Components (what exists in the repo)

### Orchestrator server (API)
- **HTTP server**: `copilot-v2/src/copilot_v2/runtime/server.py`
- **CLI entrypoint**: `copilot-v2/src/copilot_v2/scripts/orchestrator_server.py`
- Endpoints:
  - `GET /health`
  - `POST /orchestrate`

### Retrieval subsystem (“retrieval agent”)
Retrieval is hybrid (dense + lexical) and is used to map:

`goal_text` → top matching product documents → candidate `product_id`s

Code:
- `copilot-v2/src/copilot_v2/runtime/orchestrator.py` (`retrieve_evidence`)

Artifacts:
- **Owner-scoped dense index**: `copilot-v2/artifacts/indexes/<snapshot_id>/owners/<owner_id>/dense/intfloat_e5-large-v2/`

### Pricing subsystem (“pricing agent”)
Pricing provides:

`product_id` → `recommended_price_change_pct`

In demos, this is usually served from a **precomputed cache** (`pricing_cache.json`) so requests are fast.

### Sentiment subsystem (“sentiment agent”)
Sentiment provides aggregated review probabilities and a score per product:

`product_id` → `{n_reviews, p_pos, p_neu, p_neg, sentiment_score}`

In demos, this is usually served from a **precomputed cache** (`sentiment_cache.json`).

### Optional LLM debate policy (“orchestrator agent policy”)
If enabled, an LLM performs a chain-of-debate and outputs a strict JSON plan constrained to the candidate `product_id`s.

Code:
- `copilot-v2/src/copilot_v2/runtime/debate.py`
- Ollama client: `copilot-v2/src/copilot_v2/llm/ollama_client.py`

---

## Offline preparation (required for owner-scoped retrieval)

The dataset snapshot does not natively include `store_id`/`owner_id`, so we create a **deterministic synthetic owner assignment**:

- `owner_id = store_XX` computed from `product_id` via a stable hash

### 1) Build per-owner retrieval indexes

Script:
- `copilot-v2/src/copilot_v2/scripts/build_owner_indexes.py`

What it builds:
- For each owner, a FAISS dense index + corpus parquet + index metadata at:

`copilot-v2/artifacts/indexes/<snapshot_id>/owners/<owner_id>/dense/intfloat_e5-large-v2/`

Command example:

```bash
cd /path/to/cod-multiagent-ecommerce
PYTHONPATH=copilot-v2/src .venv-copilot-v2/bin/python -m copilot_v2.scripts.build_owner_indexes \
  --snapshot-id 38710839ca6e1009 \
  --artifacts-root copilot-v2/artifacts \
  --n-owners 8 \
  --device cpu
```

Notes:
- Building indexes can take time because it embeds documents.
- If you have GPU pressure, prefer CPU by passing `--device cpu`.

---

## Fast start (no rebuild): download artifacts from Google Drive

This is the recommended path for teammates: **clone the repo + download prebuilt artifacts** so they can run end-to-end **without rebuilding retrieval indexes** and **without recomputing pricing/sentiment**.

### Prerequisites

- Repo cloned locally
- Python environment created (see project setup / `.venv-copilot-v2`)
- (Optional, only if using LLM policy) Ollama running with the debate model pulled (recommended: `qwen2.5:7b-instruct`)

### Required downloads (place these files before running)

Download the following from Google Drive and unzip into the **repo root** so paths match exactly:

- **Owner retrieval indexes** (required)
- **Destination**: `copilot-v2/artifacts/indexes/38710839ca6e1009/owners/`

- **Grounding caches** (recommended for fast demos)
- **Destination**: `copilot-v2/artifacts/evals/38710839ca6e1009/orchestrator_demo/cache/`
- **Must include**:
  - `pricing_cache.json`
  - `sentiment_cache.json`

- **Model artifacts** (optional if you only use caches)
- **Pricing winner**:
  - `copilot-v2/artifacts/models/38710839ca6e1009/pricing/tabpfn/model.tabpfn_fit`
- **Sentiment winner**:
  - `copilot-v2/artifacts/models/38710839ca6e1009/sentiment/distilroberta-base_final_500k_slice_winner/`

### Where to unzip (critical)

Unzip archives into the **repository root** so that paths match exactly.

After unzip, these should exist:
- `copilot-v2/artifacts/indexes/38710839ca6e1009/owners/store_00/dense/intfloat_e5-large-v2/index_meta.json`
- `copilot-v2/artifacts/evals/38710839ca6e1009/orchestrator_demo/cache/pricing_cache.json`
- `copilot-v2/artifacts/evals/38710839ca6e1009/orchestrator_demo/cache/sentiment_cache.json`

### Portability note (important)

The owner index `index_meta.json` files are written with **relative paths** for portability, e.g.:
- `paths.corpus_parquet = "corpus.parquet"`
- `paths.faiss_index = "index_flatip.faiss"`

This means teammates can unzip anywhere, as long as the relative layout under `copilot-v2/artifacts/indexes/...` is preserved.

### Verification commands

```bash
# from repo root
ls copilot-v2/artifacts/indexes/38710839ca6e1009/owners | head
ls copilot-v2/artifacts/evals/38710839ca6e1009/orchestrator_demo/cache
```

If you need to rewrite older indexes that used absolute paths in `index_meta.json`, run:

```bash
PYTHONPATH=copilot-v2/src .venv-copilot-v2/bin/python -m copilot_v2.scripts.rewrite_index_meta_paths \
  --indexes-root copilot-v2/artifacts/indexes/38710839ca6e1009/owners
```

### Google Drive links

Add your shared links here (project-specific):
- Owner retrieval indexes: `<PASTE_LINK_HERE>`
- Grounding caches: `<PASTE_LINK_HERE>`
- Model artifacts: `<PASTE_LINK_HERE>`

---

## Runtime workflow (request → response)

### Step 1) Client calls `POST /orchestrate`
The request must include:
- `goal` (string)
- `owner_id` (string, e.g. `store_00`)

If `owner_id` is missing:
- server returns `{ "ok": false, "error": "missing_owner_id" }`

If the owner’s index is missing:
- server returns `{ "ok": false, "error": "missing_owner_index", "owner_id": "<...>" }`

### Step 2) Server routes to owner-scoped retriever
The server loads (and caches) the retriever from:

`copilot-v2/artifacts/indexes/<snapshot_id>/owners/<owner_id>/dense/intfloat_e5-large-v2/index_meta.json`

This ensures retrieval happens **within the owner’s catalog**.

### Step 3) Retrieval produces candidates + evidence
Retrieval uses:
- dense FAISS similarity on E5 embeddings
- lexical TF-IDF similarity
- a weighted hybrid score

Output is attached as evidence, e.g.:
- `evidence.retrieval_score`
- `evidence.points[]` includes a `retrieval_doc` snippet

### Step 4) Grounding enrichment attaches signals
For each candidate, the orchestrator attaches:

- **Pricing**:
  - from `pricing_cache.json` when enabled/available (`pricing.source = "cache"`)
- **Sentiment**:
  - from `sentiment_cache.json` when enabled/available
- **Inventory**:
  - reads `on_hand_units` and `safety_stock_units`
  - derives `available_to_sell = max(on_hand_units - safety_stock_units, 0)`
  - inventory agent classifies stock status: `stockout_risk | low_stock | overstocked | healthy`
- **Revenue/returns** (from synthetic sales):
  - reads `sales_daily.parquet` and aggregates to `mean_daily_revenue`, `total_returns`

### Step 5) Baseline plan is produced (grounded, deterministic)
The orchestrator produces `baseline_ranked_actions`:
- ranked candidate `product_id`s
- with `recommended_price_change_pct`, `sentiment`, `signals`, and `evidence`

### Step 6) Optional LLM debate refines the plan
If `use_llm_policy=true`, the system runs a debate policy to refine the baseline plan.

Supported modes:

- **`debate_mode = "acj"` (default)**: Advocate → Critic → Judge
  - Advocate proposes the strongest plan given retrieval + pricing + sentiment + inventory signals.
  - Critic challenges the plan and highlights risks/conflicts.
  - Judge produces the final `ranked_actions` JSON.
- **`debate_mode = "legacy"`**: specialists → peer review → judge (older pipeline)

ACJ **prompt variant** (do not mix with model-grid runs; compare in separate replays):

- **`prompt_style`** / **`prompt_version`**: same meaning as debate replay (`zero_shot_json`, `few_shot_json`, `cot_hidden`, `structured_rationale`).
- **Server defaults (demo):** `few_shot_json` + `v1` unless overridden by the request body or environment:
  - `COPILOT_V2_ACJ_PROMPT_STYLE` (default `few_shot_json`)
  - `COPILOT_V2_ACJ_PROMPT_VERSION` (default `v1`)
- Request body keys `prompt_style` and `prompt_version` override the environment for that call.

Essential safety rules:
- LLM may only use the candidate `product_id`s provided
- output must be strict JSON and is validated
- duplicates are removed during validation
- if judge output can’t be validated after retries, the system falls back to the grounded baseline plan (and records that in trace)

### Step 7) Final response JSON returned
Response includes:
- `baseline_ranked_actions` (grounded baseline)
- `ranked_actions` (final; baseline or LLM-refined)
- `trace` (includes `snapshot_id`, `owner_id`, cache wiring info)
- optional `debate_trace` (debug + timings)

---

## “Essential rules” (quick reference)

- **Owner scoping**:
  - `owner_id` is required on every request.
  - Retrieval is executed against **owner-specific indexes**.
- **Grounding**:
  - The system does not invent new SKUs; it selects from retrieved candidates.
  - Evidence + signals travel with each action.
- **LLM debate**:
  - JSON-only output enforced with parsing + validation + retries.
  - Fallback to baseline exists to keep the system robust.

---

## Debugging / reproduction tips

- Check server health:

```bash
curl -s http://127.0.0.1:8008/health
```

- Example request (baseline only):

```bash
curl -sS -X POST http://127.0.0.1:8008/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{"owner_id":"store_00","goal":"increase revenue while protecting margin","use_llm_policy":false}' \
  | python3 -m json.tool
```

- Example request (LLM policy enabled):

```bash
curl -sS -X POST http://127.0.0.1:8008/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{"owner_id":"store_00","goal":"increase revenue while protecting margin","use_llm_policy":true,"debate_mode":"acj","advocate_model":"qwen2.5:7b-instruct","critic_model":"qwen2.5:7b-instruct","judge_model":"qwen2.5:7b-instruct","candidate_m":10,"debate_top_k":3}' \
  | python3 -m json.tool
```

---

## Debate replay evaluation (debate-layer isolation)

To compare debate model combinations fairly:

1) Cache specialist outputs once (baseline pipeline only):

```bash
PYTHONPATH=copilot-v2/src .venv-copilot-v2/bin/python -m copilot_v2.scripts.cache_specialist_outputs \
  --snapshot-id 38710839ca6e1009 \
  --owner-id store_00 \
  --goals-json copilot-v2/artifacts/evals/38710839ca6e1009/llm_policy_benchmark_goals_120.json \
  --enable-pricing --enable-sentiment
```

2) Replay debate only over the saved inputs:

```bash
PYTHONPATH=copilot-v2/src .venv-copilot-v2/bin/python -m copilot_v2.scripts.replay_debate_models \
  --inputs-jsonl copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/<RUN_ID>/specialist_inputs.jsonl \
  --out-dir copilot-v2/artifacts/evals/38710839ca6e1009/debate_replay/<RUN_ID>/replay_results \
  --advocate-models llama3.1:8b,mistral:7b-instruct-v0.3-q4_K_M \
  --critic-models llama3.1:8b,mistral:7b-instruct-v0.3-q4_K_M \
  --judge-models qwen2.5:7b-instruct
```

Outputs:
- `rows.jsonl`: per-run rubric + latency
- `summary.json`: aggregated rubric means and success rate by (advocate|critic|judge) combo

---

## Where the “winners” are documented

- Full training record and rationale:
  - `copilot-v2/docs/MODEL_TRAINING_AND_RESULTS.md`
- Registry (final pointers/paths used by runtime):
  - `copilot-v2/artifacts/registry/registry.json`

