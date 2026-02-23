# Data Preparation Plan

## Stage 1: Ingestion
- Pull source datasets from Hugging Face and other public sources.
- Record source URI, version, and row counts.
- Load raw tables to BigQuery staging dataset.

## Stage 2: Cleaning
- Drop duplicates by stable keys.
- Normalize text fields (trim, lowercase copy, unicode cleanup).
- Standardize price/currency where available.
- Validate timestamp format and timezone assumptions.

## Stage 3: Canonical Modeling
- Produce canonical tables:
  - `products`
  - `reviews`
  - `queries`
  - `agent_candidates`
  - `evaluation_runs`

## Stage 4: Feature Engineering
- Build `product_document` field: title + description + category tags.
- Build review aggregates: avg rating, polarity distribution, aspect hints.
- Build price/value features: percentile band, z-score within category.

## Stage 5: Splits and Leakage Control
- Prefer temporal split when timestamps exist.
- Else use stratified split with fixed random seed.
- Keep a holdout evaluation set untouched during model tuning.

## Stage 6: Retrieval Artifacts
- Generate embeddings for products.
- Build FAISS index.
- Save lookup table to map FAISS ids to product ids.

