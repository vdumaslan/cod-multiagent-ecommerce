# Multi-Agent BI Dashboard UI

React + Tailwind frontend for the multi-agent BI demo.  
This UI calls a local Python API (`copilot_v2.apps.ui_api`) that simulates agent/model behavior with placeholder outputs.

## What this demo shows

- Query input and loading flow
- Agent pipeline execution:
  - Retrieval first
  - Sentiment, Pricing, Inventory in parallel
- Debate loop:
  - Advocate -> Critic -> Human decision (`Add Context`, `Another Round`, `Move On`) -> Judge
- Ranked plans and simulated "save to BigQuery"

## Architecture at a glance

- UI app: `copilot-v2/src/ui/src/App.jsx`
- Python API server: `copilot-v2/src/copilot_v2/apps/ui_api/server.py`
- Python API entrypoint: `copilot-v2/src/copilot_v2/apps/ui_api_server.py`
- Agent modules:
  - `retrieval_agent.py`
  - `sentiment_agent.py`
  - `pricing_agent.py`
  - `inventory_agent.py`
  - `advocate_agent.py`
  - `critic_agent.py`
  - `judge_agent.py`
- Plan persistence placeholder:
  - `plan_store.py` (`save_plan_to_bigquery`)

## Prerequisites

- Node.js LTS (includes `npm`)
- Python 3.10+ (`py` launcher or `python` in PATH)

Install Node on Windows (recommended):

```powershell
winget install OpenJS.NodeJS.LTS
```

Verify:

```powershell
node -v
npm -v
py -V
```

## Run locally (2 terminals)

### Terminal 1: Python UI API

From `copilot-v2`:

```powershell
cd "C:\Users\Victor Dumaslan\Documents\GitHub\cod-multiagent-ecommerce\copilot-v2"
$env:PYTHONPATH="src"
py -m copilot_v2.apps.ui_api_server
```

Expected startup output includes host `127.0.0.1` and port `8010`.

### Terminal 2: React UI

From `copilot-v2/src/ui`:

```powershell
cd "C:\Users\Victor Dumaslan\Documents\GitHub\cod-multiagent-ecommerce\copilot-v2\src\ui"
npm install
npm run dev
```

Open the Vite URL shown in terminal (usually `http://localhost:5173`).

## Health badge behavior

The header badge checks `http://127.0.0.1:8010/health` every 5s:

- `API checking` -> initial status
- `API online` -> Python server reachable
- `API offline` -> Python server not reachable

## API endpoints used by the UI

- `GET /health`
- `POST /ui/retrieval`
- `POST /ui/sentiment`
- `POST /ui/pricing`
- `POST /ui/inventory`
- `POST /ui/debate/advocate`
- `POST /ui/debate/critic`
- `POST /ui/debate/judge`
- `POST /ui/save_plan`

## Common issues

- **`npm` not recognized**
  - Install Node LTS, then restart terminal/IDE.
- **`No module named copilot_v2`**
  - Run from `copilot-v2` and set `PYTHONPATH`:
    - `$env:PYTHONPATH="src"`
- **Health badge stays offline**
  - Confirm Python API server is still running on port `8010`.

## Build

```powershell
npm run build
npm run preview
```

## Notes

- Current outputs are mock/placeholder values.
- No real BigQuery write is happening yet; `plan_store.py` is the extension point for real persistence.
