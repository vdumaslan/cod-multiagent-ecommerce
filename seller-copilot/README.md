# CoD Multi-Agent E-Commerce (Seller Copilot)

This directory contains the seller copilot implementation for the project.

## Goals
- Keep the entire project free (no paid APIs/subscriptions).
- Run cloud-first with BigQuery + GitHub Actions + Jupyter workflows.
- Use a debate architecture with 5 agents.
- Use at least 4 distinct models.
- Use distinct LLMs for Discovery, Sentiment, Ranking, Pricing, and Orchestrator synthesis.
- CI note: any push touching `seller-copilot/` triggers the pipeline workflow.

## Build Order
1. Lock constraints and metrics (`docs/01_constraints_and_success.md`).
2. Finalize model and dataset mapping (`docs/02_model_dataset_decision_matrix.md`).
3. Ingest and prepare data (`src/pipelines/`).
4. Train/evaluate models, including pricing FT-Transformer (`src/training/`).
5. Wire agent debate flow (`src/agents/`).
6. Launch web app (`src/app/streamlit_app.py`).

## Quickstart
```bash
pip install -r seller-copilot/requirements.txt
```

Run pipeline once (idempotent rerun-safe):
```bash
python seller-copilot/src/pipelines/run_pipeline.py --config seller-copilot/config/pipeline.yaml
```

Pipeline output:
- BigQuery stage tables:
  - `stg_amazon_reviews`
  - `stg_amazon_meta`
  - `stg_twitter_support`
  - `stg_online_retail`
  - `stg_telco_churn`
- BigQuery canonical/model-ready tables:
  - `products`, `reviews`, `support_tickets`, `retail_transactions`, `churn_signals`
  - `product_features`, `sentiment_dataset`, `ranking_pairs`, `pricing_features`, `retrieval_corpus`, `training_splits`
- BigQuery run logs:
  - `pipeline_runs`
- Local artifact:
  - `seller-copilot/artifacts/quality_report.json`

Scale for model training:
- Increase `pipeline.max_rows_per_source` and/or per-source `target_rows` in `seller-copilot/config/pipeline.yaml`.
- For validation smoke tests, pass `--max-rows 5000` or `--max-rows 10000`.

For local/manual runs only, set environment variables:
```bash
setx GOOGLE_APPLICATION_CREDENTIALS "C:\Users\niran\OneDrive\Documents\linear-theater-436300-r9-f04051db9e69.json"
setx GCP_PROJECT_ID "linear-theater-436300-r9"
setx BIGQUERY_LOCATION "US"
setx BIGQUERY_DATASET "seller_copilot_prod"
```

## Cloud Scheduling (GitHub Actions, free)
The pipeline is scheduled daily and can be triggered manually from:
- `.github/workflows/seller-copilot-pipeline.yml`

Configure repository settings:
- `Secrets`:
  - `GCP_SA_KEY_JSON` = full service account JSON content
  - `HF_TOKEN` (optional) = Hugging Face token for higher API rate limits on larger runs
- `Variables` (optional; defaults already exist in workflow):
  - `GCP_PROJECT_ID` = `linear-theater-436300-r9`
  - `BIGQUERY_LOCATION` = `US`
  - `BIGQUERY_DATASET` = `seller_copilot_prod`

Then run via:
- GitHub -> Actions -> `Seller Copilot Data Pipeline` -> `Run workflow`
- Or wait for the daily cron schedule.

Run app:
```bash
streamlit run seller-copilot/src/app/streamlit_app.py
```

LLM runtime notes:
- Agent LLM IDs are configured in `seller-copilot/config/models.yaml`.
- The app uses Hugging Face Inference API when `HF_TOKEN` is set.
- If an LLM endpoint is unavailable, agents fall back to deterministic claims based on model evidence.

