# CoD Multi-Agent E-Commerce (Seller Copilot)

This folder is a clean restart and does not depend on existing repository code.

## Goals
- Keep the entire project free (no paid APIs/subscriptions).
- Run cloud-first with BigQuery + Prefect + Jupyter workflows.
- Use a debate architecture with 5 agents.
- Use at least 4 distinct models.

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

Run pipeline flow:
```bash
python seller-copilot/src/pipelines/prefect_flow.py
```

Pipeline output:
- BigQuery tables: `stg_amazon_reviews`, `products`, `reviews`, `product_features`
- Local artifact: `seller-copilot/artifacts/quality_report.json`

Run app:
```bash
streamlit run seller-copilot/src/app/streamlit_app.py
```

