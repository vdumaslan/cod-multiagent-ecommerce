# copilot-v2 / app

This directory contains the runtime application layer for the seller copilot — the specialist agents, debate orchestrator, API, and UI.

---

## Directory Structure

```
app/
├── agents/                  # Specialist agents (runtime, reads from cache)
│   ├── retrieval_agent.py   # Dense retrieval using intfloat/e5-large-v2 + FAISS
│   ├── pricing_agent.py     # Pricing recommendations from TabPFN cache
│   ├── sentiment_agent.py   # Sentiment signals from DistilRoBERTa cache
│   ├── inventory_agent.py   # Stock classification from rule-based cache
│   └── orchestrator/        # Debate layer (Advocate, Critic, Judge)
│       ├── advocate.py
│       ├── critic.py
│       ├── judge.py
│       ├── human_review.py
│       └── orchestrator.py
├── api/                     # FastAPI server
│   ├── app.py
│   └── schemas.py
├── ui/                      # React frontend
├── pipeline.py              # Main entry point wiring all agents together
└── APP_IMPLEMENTATION.md    # Detailed implementation notes and known gaps
```

**Precompute scripts** live in `src/copilot_v2/scripts/precompute/` — not in this directory.

---

## How It Works

The system has two phases:

**Offline (precompute) — run once**
```
Raw parquet files → src/copilot_v2/scripts/precompute/ → caches in artifacts/caches/
```

**Runtime — on every user query**
```
User query → retrieval agent → specialist agents (pricing, sentiment, inventory)
           → debate (Advocate → Critic → Judge) → ranked action plan
```

The specialist agents read only from pre-built caches at runtime — no model inference except for encoding the user query in the retrieval agent.

---

## Debate Models

| Role | Model |
|---|---|
| Advocate | `llama3.1:8b` (via Ollama) |
| Critic | `qwen2.5:7b-instruct` (via Ollama) |
| Judge | `qwen2.5:7b-instruct` (via Ollama) |
| Prompt style | `few_shot_json` + `v1` |

Set via env vars: `COPILOT_V2_ACJ_PROMPT_STYLE`, `COPILOT_V2_ACJ_PROMPT_VERSION`

---

## Running the Precompute Pipeline

All precompute scripts are in `src/copilot_v2/scripts/precompute/`. Run from the repo root:

```bash
# Step 1: Generate pricing labels
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_pricing_training_table \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts

# Step 2: Build pricing cache (requires TabPFN model artifact + step 1 output)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_pricing_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 3: Build sentiment cache (DistilRoBERTa model)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_sentiment_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach distilroberta --write-json
# Fast fallback (star ratings, no model needed):
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_sentiment_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach ratings

# Step 4: Build inventory cache (rule-based, no model needed)
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_inventory_cache \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 5: Build per-owner retrieval indexes
PYTHONPATH=copilot-v2/src python -m copilot_v2.scripts.precompute.build_owner_indexes \
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --device cpu
```

The retrieval FAISS index may already be built at `artifacts/indexes/38710839ca6e1009/` — check before running step 5.

---

## Known Gaps

See `APP_IMPLEMENTATION.md` for full details. Summary:

| Gap | Status |
|---|---|
| `recommended_price_change_pct` | Not missing — computed on demand by `build_pricing_training_table.py` |
| Inventory cache | Run `build_inventory_cache.py` once to generate |
| Agents not wired to `pipeline.py` | Pending |
