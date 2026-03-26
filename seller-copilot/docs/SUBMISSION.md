# Submission checklist

## 1) Stop a stuck run

If `generate_data.py` is still running in a terminal: **Ctrl+C**.

## 2) Generate data (minutes)

```powershell
python seller-copilot/scripts/generate_data.py --quick
```

Or:

```powershell
python seller-copilot/scripts/generate_data.py --products 8000 --pool-reviews 12000 --pool-meta 8000
```

Then:

```powershell
python seller-copilot/scripts/validate_agent_data.py
```

## 3) BigQuery (optional)

```powershell
python seller-copilot/scripts/upload_to_bigquery.py --reset
```

See `docs/GCP_SETUP.md` and `seller-copilot/.env`.

## 4) Models / comparison / demo (short plan)

| Piece | Fast approach |
|-------|----------------|
| Sentiment | VADER + one transformer on a sample of `reviews.parquet`; tabulate F1. |
| Tabular | CatBoost defaults on joined products + signals; MAE on held-out price. |
| Retrieval | Embed `retrieval_corpus` subset, FAISS, Recall@5. |
| Demo | One Streamlit page: load Parquet, show goal + top products + snippets. |

## 5) Report

Dataset stats, model table, Streamlit screenshots, limitations.
