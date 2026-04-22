# copilot-v2 / app

This directory contains the runtime application layer for the seller copilot — the specialist agents, debate orchestrator, precompute pipeline, API, and UI.

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
├── precompute/              # Offline jobs — run once to build caches
│   ├── build_pricing_features.py   # (not yet active) joins raw files for full coverage
│   ├── precompute_pricing.py       # TabPFN inference → pricing cache
│   ├── precompute_sentiment.py     # DistilRoBERTa inference → sentiment cache
│   └── precompute_inventory.py     # Rule-based classification → inventory cache
├── api/                     # FastAPI server
│   ├── app.py
│   └── schemas.py
├── ui/                      # React frontend
├── pipeline.py              # Main entry point wiring all agents together
└── APP_IMPLEMENTATION.md    # Detailed implementation notes and known gaps
```

---

## How It Works

The system has two phases:

**Offline (precompute) — run once**
```
Raw parquet files → precompute scripts → caches in artifacts/caches/
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

Run these once before starting the server. The inventory cache does not exist yet and must be generated first.

```bash
# Inventory — rule-based, no model needed, run this first
python -m app.precompute.precompute_inventory

# Pricing — requires artifacts/models/.../tabpfn/model.tabpfn_fit.zip
python -m app.precompute.precompute_pricing

# Sentiment — requires artifacts/models/.../distilroberta-base_final_500k_slice_winner/
python -m app.precompute.precompute_sentiment

# Sentiment fast fallback (uses star ratings instead of model, for testing only)
python -m app.precompute.precompute_sentiment --no-model
```

The retrieval FAISS index is already built and saved at `artifacts/indexes/38710839ca6e1009/dense/intfloat_e5-large-v2/` — no precompute step needed for retrieval.

---

## Known Gaps

See `APP_IMPLEMENTATION.md` for full details. Summary:

| Gap | Status |
|---|---|
| Pricing cache covers ~50k products only | Sent to team — needs feature pipeline owner to validate `build_pricing_features.py` |
| `recommended_price_change_pct` target missing | Sent to team — needed if TabPFN ever needs retraining |
| Inventory cache not yet generated | Run `precompute_inventory.py` once |
| Agents not wired to `pipeline.py` | Pending |
