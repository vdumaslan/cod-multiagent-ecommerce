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

## Training / tuning entrypoints (recreated)

These scripts generate the **same artifact layouts** referenced by `copilot-v2/docs/MODEL_TRAINING_AND_RESULTS.md`.

- **Sentiment fine-tune (HF Trainer)**
  - Script: `copilot-v2/src/copilot_v2/scripts/train_sentiment_encoder.py`
  - Writes:
    - Model checkpoints under `copilot-v2/artifacts/models/<snapshot_id>/sentiment/...`
    - Metrics under `copilot-v2/artifacts/evals/<snapshot_id>/sentiment/trial_runs/<run_tag>/`

- **Retrieval dense tuning (val; screen + refine)**
  - Script: `copilot-v2/src/copilot_v2/scripts/tune_retrieval_dense.py`
  - Writes: `copilot-v2/artifacts/evals/<snapshot_id>/retrieval/tuning/tune_report*.json`

- **Pricing labels + training table**
  - Script: `copilot-v2/src/copilot_v2/scripts/build_pricing_training_table.py`
  - Writes:
    - `pricing_labels.parquet`
    - `pricing_training_table.parquet`
    - `pricing_training_report.json`
    - `pricing_label_recipe.json`

- **Pricing tune (TabPFN / CatBoost)**
  - Scripts:
    - `copilot-v2/src/copilot_v2/scripts/tune_pricing_tabpfn.py`
    - `copilot-v2/src/copilot_v2/scripts/tune_pricing_catboost.py`
  - Writes: `tabpfn_tune_report.json`, `catboost_tune_report.json`

- **Pricing tune (FT-Transformer)**
  - Script: `copilot-v2/src/copilot_v2/scripts/tune_pricing_ft_transformer.py`
  - Note: entrypoint + report schema are recreated; full training loop is not yet implemented in `copilot-v2`.

## What lives where (high level)

- **Runtime**
  - `copilot-v2/src/copilot_v2/runtime/`
- **Scripts**
  - `copilot-v2/src/copilot_v2/scripts/`
- **Docs**
  - `copilot-v2/docs/`
- **Artifacts (large)**
  - `copilot-v2/artifacts/`
