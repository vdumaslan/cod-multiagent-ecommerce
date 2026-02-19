---
name: data-pipeline
description: Use this skill when working with data ingestion, transformation, preprocessing, embeddings, parquet files, the cloud pipeline, or any of the existing notebooks in this project
---

# Data Pipeline
Cloud-based pipeline fully built.  Serves as the foundation for all agent development and the debate system.  Responsible for ingesting raw data, transforming it into a usable format, and storing it efficiently for downstream use by the agents.

## Infrastructure
- **Cloud Provider:** TBD (AWS/GCP/Azure)
- All datasets stored in Parquet format (~5x faster load than CSV, ~60% smaller file size)
- **Embedding model:** sentence-transformers MiniLM-L6-v2 (In notebook 04)
- **Vector Search:** FAISS index for semantic candidate retrieval
- **ETL Orchestration:** Python-based pipeline scripts

# Data Source
Amazon Reviews 2023 (Hugging Face: McAuley-Lab/Amazon-Reviews-2023)
- Categories used: All_Beauty, Electronics, Books (50,000 reviews each = 150,000 total)
- After cleaning: 133,730 reviews (16,270 duplicates removed)
- Used for: Sentiment Agent, Ranking Agent, Pricing Agent

## Key Processed Files (data/processed/)
- amazon_sentiment_cleaned.parquet — 133,730 cleaned reviews (for Sentiment Agent)
- amazon_product_meta_cleaned.parquet — 90,000 products (title, category, price, avg_rating)
- amazon_pricing_ready.parquet — 90,000 products with review-level aggregations
- product_intelligence_train/val/test.parquet — 72K/9K/9K stratified splits
- pricing_train/val/test.parquet — 72K/9K/9K stratified splits

## Train/Val/Test Strategy
- 80/10/10 stratified by category (equal 33.33% each: All_Beauty, Electronics, Books)
- Random state: 42 (reproducibility)

## Pipeline Stages (notebooks-final/)
1. 01_data_ingestion.ipynb — raw data ingestion
2. 02_data_transformation.ipynb — cleaning and transforming
3. 03_data_warehouse_structure.ipynb — data warehouse setup
4. 04_embedding_pipeline.ipynb — MiniLM-L6-v2 embeddings + FAISS index
5. 05_sentiment_agent_setup.ipynb — Sentiment Agent (IN PROGRESS)
6. 06_LLM_Agents.ipynb — all agents + debate logic (PLANNED)
7. 07_pipeline_automation.ipynb — full automation (PLANNED)

## Retrieval Flow (feeds into Discovery Agent)
User query → MiniLM-L6-v2 → query embedding → FAISS top-K → candidates passed to agents

## Security
- Secrets in .env files, never committed to Git
- Data encrypted at rest and in transit