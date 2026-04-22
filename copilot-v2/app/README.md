# copilot-v2 / app

This directory contains the runtime application layer for the seller copilot — the specialist agents, debate orchestrator, API, and React UI.

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
├── ui/                      # React + Vite frontend
├── llm.py                   # OllamaClient and JSON schema validators
├── pipeline.py              # Main entry point wiring all agents together
└── APP_IMPLEMENTATION.md    # Detailed implementation notes
```

**Precompute scripts** live in `src/copilot_v2/scripts/precompute/` — not in this directory.

---

## How It Works

The system has two phases:

**Offline (precompute) — run once**
```
Raw parquet files → src/copilot_v2/scripts/precompute/ → caches in artifacts/caches/
```

**Runtime — on every user query (4 API calls)**
```
POST /pipeline  (fast, ~2s)
  User query → retrieval agent → pricing + sentiment + inventory agents
  → returns enriched candidates + baseline ranked actions

POST /debate/start  (slow, LLMs)
  enriched candidates → Advocate LLM → Critic LLM (round 1)
  → returns advocate + critic outputs

POST /debate/continue  (slow, LLMs — repeated N times, user-controlled)
  prev_advocate + prev_critic → Advocate (revision) → Critic (revision)
  → returns updated advocate + critic

POST /debate/judge  (slow, LLMs — once, when user clicks "Move On")
  latest_advocate + latest_critic → Judge LLM
  → returns final ranked action plan
```

Round decisions (continue vs. move on) are owned by the UI. The backend runs exactly what it is asked.

The specialist agents read only from pre-built caches — no model inference except for encoding the user query in the retrieval agent.

---

## Debate Models

| Role | Model | Via |
|---|---|---|
| Advocate | `llama3.1:8b` | Ollama |
| Critic | `qwen2.5:7b-instruct` | Ollama |
| Judge | `qwen2.5:7b-instruct` | Ollama |

Prompt style: `few_shot_json` + `v1`

---

## Setup

### Prerequisites

**1. Python environment**

```bash
# From repo root — activate the existing venv if present
.venv-copilot-v2\Scripts\activate   # Windows
source .venv-copilot-v2/bin/activate # Mac/Linux

pip install -r copilot-v2/requirements.txt
```

**2. Node.js** (for the UI)

```bash
cd copilot-v2/app/ui
npm install
```

**3. Ollama**

Download and install from https://ollama.com. After installation, pull the two models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
```

Verify Ollama is running: open `http://localhost:11434` — it should return `Ollama is running`.

On Windows, Ollama runs as a background service after install (check the system tray). No manual `ollama serve` needed.

---

## Building the Caches (one-time setup)

Run these from the repo root in order. Check `artifacts/caches/38710839ca6e1009/` first — pricing and sentiment caches may already exist.

```powershell
# Activate venv first
$env:PYTHONPATH = "copilot-v2/src"

# Step 1: Generate pricing labels
python -m copilot_v2.scripts.precompute.build_pricing_training_table `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts

# Step 2: Build pricing cache (requires TabPFN model artifact + step 1 output)
python -m copilot_v2.scripts.precompute.build_pricing_cache `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 3: Build sentiment cache (DistilRoBERTa model)
python -m copilot_v2.scripts.precompute.build_sentiment_cache `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach distilroberta --write-json
# Fast fallback (star ratings, no model needed):
python -m copilot_v2.scripts.precompute.build_sentiment_cache `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --approach ratings

# Step 4: Build inventory cache (rule-based, always fast)
python -m copilot_v2.scripts.precompute.build_inventory_cache `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --write-json

# Step 5: Build retrieval index (check artifacts/indexes/ first — may already exist)
python -m copilot_v2.scripts.precompute.build_owner_indexes `
  --snapshot-id 38710839ca6e1009 --artifacts-root copilot-v2/artifacts --device cpu
```

---

## Running the System

Three things to start (two terminals + Ollama in background):

**Terminal 1 — API server** (from repo root):

```powershell
cd copilot-v2
$env:COPILOT_ARTIFACTS_ROOT = "$PWD\artifacts"
$env:PYTHONPATH = "$PWD"
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — React UI:**

```bash
cd copilot-v2/app/ui
npm run dev
```

Open `http://localhost:5173` in your browser.

**Verify the API is healthy:**

```powershell
curl http://localhost:8000/health -UseBasicParsing
```

Expected: `{"ok":true,"has_pricing_cache":true,"has_sentiment_cache":true,"has_inventory_cache":true}`

If any cache shows `false`, run the corresponding precompute step above.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Cache load status for all three specialist agents |
| `POST` | `/pipeline` | Retrieval + enrichment + baseline (fast, no LLMs) |
| `POST` | `/debate/start` | Advocate + Critic round 1 (LLMs, slow) |
| `POST` | `/debate/continue` | One more Advocate + Critic revision round (LLMs) |
| `POST` | `/debate/judge` | Judge once on final debate; returns ranked actions |
| `POST` | `/orchestrate` | Legacy all-in-one endpoint (kept for compatibility) |

Env vars:
- `COPILOT_SNAPSHOT_ID` (default: `38710839ca6e1009`)
- `COPILOT_ARTIFACTS_ROOT` (default: `copilot-v2/artifacts`)
- `COPILOT_OLLAMA_URL` (default: `http://localhost:11434`)

---

## Artifact Saving

Every API call writes outputs to a timestamped folder under `artifacts/runs/{snapshot_id}/`. Files are written immediately after each agent completes.

| Folder suffix | Created by | Files |
|---|---|---|
| `run_{ts}_{owner}/` | `/pipeline` | `1_retrieval.json`, `2_enriched.json`, `3_baseline.json` |
| `run_{ts}_{owner}_debate/` | `/debate/start` | `4_debate_advocate_r1.json`, `5_debate_critic_r1.json` |
| `run_{ts}_{owner}_cont/` | `/debate/continue` | `4_debate_advocate.json`, `5_debate_critic.json` |
| `run_{ts}_{owner}_judge/` | `/debate/judge` | `8_debate_judge.json`, `9_final.json` |

---

## Known Gaps

See `APP_IMPLEMENTATION.md` for full details.

| Item | Status |
|---|---|
| Inventory cache | Run `build_inventory_cache.py` once — not shipped in repo |
| `recommended_price_change_pct` | Not missing — generated by `build_pricing_training_table.py` |
| Cloud LLM migration | Requires prompt re-validation — prompts were tuned for llama3.1:8b + qwen2.5:7b |
