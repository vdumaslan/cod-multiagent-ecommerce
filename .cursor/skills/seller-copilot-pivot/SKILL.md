---
name: seller-copilot-pivot
description: Capture current project direction for Seller Copilot. Use when changing agents, pipelines, datasets, retrieval, training, or evaluation. Enforces pivot rules: minimize LLM usage and replace Amazon sources with Unwrangle Costco API.
---

# Seller Copilot Pivot (Costco + minimal LLM)

## Current pivot rules (must follow)

- **LLMs are optional**: do not introduce LLM calls for an agent unless the user explicitly asks. Prefer deterministic/statistical logic and small local models first.
- **No Amazon datasets**: do not add or depend on Amazon Reviews 2023 / Amazon metadata ingestion. Replace dataset assumptions throughout the pipeline and training code.
- **Primary data source is Costco via Unwrangle API**: treat Unwrangle’s Costco endpoints as the canonical product/review/price feed going forward.

## When working on this repo, default approach

1. **Start from the pipeline**: update `seller-copilot/config/pipeline.yaml` + `seller-copilot/src/pipelines/ingest_sources.py` first, then BigQuery SQL transforms/features, then training artifacts, then agents/app.
2. **Prefer end-to-end shape compatibility**: keep the canonical tables (`products`, `reviews`, `product_features`, etc.) consistent so the agents/app keep working while the data source changes.
3. **Avoid paid / token-billed dependencies**: any new dependency must be free to run locally and in GitHub Actions.

## Agent policy (default)

- **Discovery**: can be pure retrieval (embeddings + FAISS) with no LLM narrative; return evidence + candidates.
- **Sentiment**: prefer classifier outputs or dataset-derived aggregates; LLM narration off by default.
- **Ranking**: prefer deterministic scoring / reranker model; LLM narration off by default.
- **Pricing**: prefer tabular model/statistics; LLM narration off by default.
- **Orchestrator**: prefer rule-based synthesis (vote + confidence) and only use LLM for final text if requested.

## Notes to keep in mind

- `seller-copilot/src/agents/llm_runtime.py` currently uses Hugging Face Inference API and requires `HF_TOKEN`. Under this pivot, this should not be required for normal runs.
- The repo currently has BigQuery transforms/features designed around Amazon-shaped data; as Costco ingestion is implemented, update SQL schemas and downstream training scripts accordingly.

