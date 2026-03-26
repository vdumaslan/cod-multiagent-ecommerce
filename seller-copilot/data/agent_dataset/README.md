# Agent / model dataset (generated locally)

Populated either by **Stage 1** (`run_stage1.py` + local JSONL) or by **`scripts/generate_data.py`**, which samples text from the McAuley Home & Kitchen files and assembles agent-ready tables.

Parquet files are **not committed** (see repo `.gitignore`). This README is.

## Files (feed these to your stack)

| File | Role |
|------|------|
| **`products.parquet`** | Curated products: titles, descriptions, price, category, merged review stats. |
| **`reviews.parquet`** | All curated reviews for those products (NLP, sentiment, sequence models). |
| **`product_signals.parquet`** | Per-product aggregates (review counts, positive ratio, sentiment/recency scores). |
| **`retrieval_corpus.parquet`** | Slim RAG bundle: `product_id` + `product_document` (+ category, price, avg_rating). |
| **`agent_dataset_manifest.json`** | Row counts, quality-gate settings, column lists, timestamps. |

Duplicate copies for debugging live under `seller-copilot/artifacts/stage1/` (`*_curated.parquet`).

## How to regenerate

```bash
python seller-copilot/scripts/generate_data.py --quick
# or full Stage 1:
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_local_agent.yaml
```

Requires the large raw JSONL files under `seller-copilot/data/raw/amazon_reviews_2023/`.
