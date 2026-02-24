# Model + Dataset Decision Matrix

Date: 2026-02-22

## Final Model Set (Free/Open)

1. Orchestrator / Debate Synthesis  
- Model: `Qwen/Qwen2.5-7B-Instruct`  
- Why: strong open instruct model quality for structured synthesis and conflict resolution.

2. Discovery Debate Reasoning  
- Model: `mistralai/Mistral-7B-Instruct-v0.3`  
- Why: strong instruction-following quality for concise retrieval-grounded claims.

3. Retrieval Embeddings  
- Model: `BAAI/bge-large-en-v1.5`  
- Why: high retrieval quality for semantic search.

4. Candidate Reranking  
- Model: `BAAI/bge-reranker-v2-m3`  
- Why: strong reranking performance for query-product matching.

5. Sentiment / User Voice  
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- Why: robust sentiment reference model for short-form and support-style text.

6. Sentiment Debate Reasoning  
- Model: `microsoft/Phi-3.5-mini-instruct`  
- Why: compact instruct LLM for evidence-to-claim synthesis.

7. Ranking Debate Reasoning  
- Model: `meta-llama/Llama-3.1-8B-Instruct`  
- Why: strong zero-shot reasoning for pairwise and listwise recommendation arguments.

8. Pricing / Value  
- Model: `FT-Transformer`  
- Why: modern deep tabular model suited to pricing features (price bands, category context, rating aggregates, sentiment-derived features) and stronger than simple linear/rule reference models for nonlinear feature interactions.

9. Pricing Debate Reasoning  
- Model: `google/gemma-2-9b-it`  
- Why: high quality instruction model for concise value/risk narratives.

Fallback options:
- LLM fallback: `HuggingFaceH4/zephyr-7b-beta` and `Qwen/Qwen2.5-3B-Instruct`
- Embedding fallback: `sentence-transformers/all-MiniLM-L6-v2`

## Dataset Mapping

### Discovery + Ranking + Pricing
- Source: Amazon Reviews 2023 + product metadata (Hugging Face)
- Use:
  - retrieval corpus (title/description/category/price)
  - relevance pairs for ranking evaluation
  - review aggregates for pricing/value features

### Sentiment Agent
- Source 1: Amazon Reviews 2023
- Source 2: Twitter Customer Support
- Use:
  - train/adapt sentiment and aspect signals
  - create user-voice summaries per product candidate

### Predictive/Behavior Context
- Source: Online Retail II + Telco Churn
- Use:
  - customer/transaction behavior context features
  - optional retention-risk signal for orchestrator rationale

## Artifact Outputs
- `curated_products.parquet`
- `curated_reviews.parquet`
- `retrieval_index.faiss`
- `retrieval_lookup.parquet`
- `sentiment_train/val/test.parquet`
- `ranking_eval_pairs.parquet`
- `metrics_summary.json`
