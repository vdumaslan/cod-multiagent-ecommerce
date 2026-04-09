# Copilot V2 (Multi-agent Orchestrator)

This folder contains the **grounded, owner-scoped, multi-agent orchestrator** (retrieval + pricing + sentiment + debate/judge) and the scripts/docs needed to run it end-to-end.

## Start here

- **System workflow (canonical)**: `copilot-v2/docs/SYSTEM_WORKFLOW.md`
  - Includes: artifact downloads (fast start), owner-scoped retrieval indexes, runtime request/response flow, debugging, and reproduction steps.
- **Model training + winners registry**: `copilot-v2/docs/MODEL_TRAINING_AND_RESULTS.md` (+ `.docx`)

## Fast start (recommended)

Follow the **“Fast start (no rebuild)”** section in:

- `copilot-v2/docs/SYSTEM_WORKFLOW.md`

In short, teammates can **clone the repo + download artifacts (indexes + caches)** and run without rebuilding retrieval indexes or recomputing pricing/sentiment.

## What lives where (high level)

- **Runtime**
  - `copilot-v2/src/copilot_v2/runtime/`
- **Scripts**
  - `copilot-v2/src/copilot_v2/scripts/`
- **Docs**
  - `copilot-v2/docs/`
- **Artifacts (large)**
  - `copilot-v2/artifacts/`
