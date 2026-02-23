# Constraints And Success Criteria

Date: 2026-02-22

## Hard Constraints
- Budget: zero paid cost.
- APIs: no paid/token-billed model APIs.
- Cloud-first execution for storage and pipeline.
- Local GPU is optional only for model training/agent setup fallback.
- Minimum 4 distinct models in final system.
- Debate architecture is required.

## Mandatory Stack
- Data warehouse: BigQuery Sandbox.
- Orchestration: Prefect.
- Training/analysis: Jupyter notebooks or Python scripts in cloud runtime.
- Retrieval: FAISS artifact + metadata from BigQuery.
- Web app: Streamlit.

## Success Metrics
- Retrieval: Recall@10, nDCG@10, MRR.
- Ranking: nDCG@10 or pairwise accuracy.
- Sentiment: macro F1.
- Pricing: MAE/RMSE for fair-price prediction + calibration of value score.
- End-to-end: success@1, median latency, p95 latency.
- System quality: reproducibility (fixed seeds, versioned artifacts, deterministic splits).
