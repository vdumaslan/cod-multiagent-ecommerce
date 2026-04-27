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

**1. Python environment** (3.9 or later required)

```bash
# From repo root — create the venv if it doesn't exist yet
python -m venv .venv-copilot-v2

# Activate it
.venv-copilot-v2\Scripts\activate   # Windows
source .venv-copilot-v2/bin/activate # Mac/Linux

pip install -r copilot-v2/requirements.txt
```

> **Note:** `requirements.txt` includes the full ML stack (`torch`, `transformers`, `tabpfn`, `catboost`) which is only needed for the precompute scripts. If you only want to run the app and the caches/indexes are already built, the runtime-only dependencies are: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pandas`, `pyarrow`, `faiss-cpu`, `sentence-transformers`.

**2. Node.js 18 or later** (for the UI)

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

- **Windows / Mac:** Ollama runs as a background service after install (check the system tray / menu bar). No manual `ollama serve` needed.
- **Linux:** Run `ollama serve` in a separate terminal before starting the app. It needs to stay running.

---

## Required Artifacts

The app reads from these pre-built artifacts at runtime (all under `copilot-v2/artifacts/`). If you don't have them locally, download them from the shared Google Drive and place them under `copilot-v2/artifacts/`.

| Path | Used by | Notes |
|---|---|---|
| `caches/38710839ca6e1009/pricing/pricing_cache.parquet` | `pricing_agent.py` | TabPFN inference precomputed |
| `caches/38710839ca6e1009/sentiment/sentiment_cache.parquet` | `sentiment_agent.py` | DistilRoBERTa inference precomputed |
| `caches/38710839ca6e1009/inventory/inventory_cache.parquet` | `inventory_agent.py` | Rule-based, fast to regenerate |
| `indexes/38710839ca6e1009/dense/intfloat_e5-large-v2/` | `retrieval_agent.py` | FAISS index over retrieval corpus |

Everything else in `artifacts/` (`data_snapshots/`, `models/`, `features/`, `evals/`, etc.) is only needed by the precompute pipeline — not the running app.

The `/health` endpoint reports which caches loaded successfully on startup.

---

## Building the Caches (one-time setup)

Run these from the repo root in order. Check `artifacts/caches/38710839ca6e1009/` first — pricing and sentiment caches may already exist.

> **Mac/Linux:** Replace the backtick `` ` `` line-continuation character with `\` in the commands below.

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

Windows (PowerShell):
```powershell
.venv-copilot-v2\Scripts\activate
cd copilot-v2
$env:COPILOT_ARTIFACTS_ROOT = "$PWD\artifacts"
$env:PYTHONPATH = "$PWD"
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Mac/Linux:
```bash
source .venv-copilot-v2/bin/activate
cd copilot-v2
export COPILOT_ARTIFACTS_ROOT="$PWD/artifacts"
export PYTHONPATH="$PWD"
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — React UI:**

```bash
cd copilot-v2/app/ui
npm run dev
```

Open `http://localhost:5173` in your browser.

The UI polls the `/health` endpoint on load and shows one of four badge states (top-right):

| Badge | Colour | Meaning |
|---|---|---|
| **API CHECKING** | Yellow | Waiting for the first health response |
| **API ONLINE** | Green | All caches and retrieval index loaded — safe to submit |
| **API DEGRADED** | Orange | API is running but one or more components failed to load — check `/health` response and run the missing precompute step |
| **API OFFLINE** | Red | API is not reachable — check that the server is running |

> **If you submit a query and the UI silently resets back to the query page**, the badge should show **API DEGRADED**. The most common cause is the retrieval index not being present at `artifacts/indexes/38710839ca6e1009/dense/intfloat_e5-large-v2/` — download it from the shared Google Drive. If the badge shows **API ONLINE** but the reset still happens, check the API server terminal — FastAPI prints the full error traceback there.

> **First-run note:** On the very first startup, the retrieval agent downloads `intfloat/e5-large-v2` (~1.3 GB) from HuggingFace automatically. This requires internet access and will make the first startup take several minutes. The model is cached locally (in `~/.cache/huggingface/`) and subsequent startups are fast.

> **Debate latency:** The `/debate/start`, `/debate/continue`, and `/debate/judge` steps each call Ollama LLMs and typically take **30–90 seconds** per step depending on your hardware. This is expected — the UI shows a loading state while waiting.

> **Mock data fallback:** If the debate steps fail silently (e.g. Ollama times out or returns invalid JSON), the UI will display hardcoded placeholder plans rather than an error. If results look generic or identical across queries, check that Ollama is running and the models are pulled.

> **Changing the API port:** The Vite dev proxy in `ui/vite.config.js` is hardcoded to `http://127.0.0.1:8000`. If you run the API on a different port, update that file to match — otherwise the UI will fail silently.

> **Owner ID:** The UI sends `store_00` as the owner identifier — this is just a label used to name run output folders and has no effect on results.

**Verify the API is healthy:**

```powershell
curl http://localhost:8000/health -UseBasicParsing
```

Expected: `{"ok":true,"has_pricing_cache":true,"has_sentiment_cache":true,"has_inventory_cache":true,"has_retrieval_index":true}`

If any cache shows `false`, run the corresponding precompute step above.

> **Note:** If a cache file is missing entirely (not just empty), the API will return a 500 error on startup rather than a graceful `false`. If you see a 500 on `/health`, it means a required cache file was not found — run the corresponding precompute step and restart the server.

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
| Inventory cache | Available on shared Google Drive — run `build_inventory_cache.py` to regenerate if needed |
| `recommended_price_change_pct` | Not missing — generated by `build_pricing_training_table.py` |
| Cloud LLM migration | Requires prompt re-validation — prompts were tuned for llama3.1:8b + qwen2.5:7b |
