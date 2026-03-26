# Business Decision Copilot: End-to-End Implementation Plan

Locked direction:

- Dataset vertical: `Amazon Reviews 2023 - Home_and_Kitchen`
- Product type: domain-configurable business decision copilot
- Core goals: revenue uplift, cost reduction, margin improvement, dead inventory reduction, pricing optimization, customer satisfaction improvement
- Core runtime pattern: RAG-grounded multi-agent debate + human approval loop

---

## 0) Scope Lock and Delivery Targets

### 0.1 In-Scope (v1)

- Owner asks business goal in natural language.
- System returns 2-3 ranked action plans with evidence and risk.
- Human approves/rejects and optional reason.
- Feedback is stored and used for preference-aware re-ranking.

### 0.2 Out-of-Scope (v1)

- Auto-executing price/inventory changes.
- Workforce/hiring optimization.
- Full LLM fine-tuning.

### 0.3 Final Deliverables

- Working Streamlit app demo.
- Reproducible data pipeline and artifacts.
- Model comparison results and final model justifications.
- Final report + appendix evidence pack.

---

## 1) Data Acquisition and Curation

### 1.1 Source

- Hugging Face dataset: `McAuley-Lab/Amazon-Reviews-2023`
- Files:
  - `raw/review_categories/Home_and_Kitchen.jsonl`
  - `raw/meta_categories/meta_Home_and_Kitchen.jsonl`

### 1.2 Ingestion Strategy

- **Local (recommended after download):** two-pass scan of on-disk JSONL — pass 1 counts qualifying reviews per `parent_asin` over the full file; pass 2 collects all reviews + meta for the top-K products, then curate and load BigQuery.
- **Remote (CI / smoke):** stream partial JSONL over HTTP with row caps.
- Stream JSONL in chunks (no full-memory load of the 43 GB files).
- Parse and normalize product/review fields.
- Store raw stage tables first.

### 1.3 Quality Gates (must pass before indexing/training/eval)

- Required IDs and non-empty title/text fields.
- Valid or recoverable prices.
- De-duplication by stable IDs.
- Review count threshold per product (start at `>=15`).
- Recency threshold (keep recent active products).
- Category/subcategory balance quotas.
- Outlier handling for price/rating by subcategory.

### 1.4 Target Subset Size

- Products: `50k-150k`
- Reviews: `500k-2M`
- Keep only reviews linked to selected products.

### 1.5 Synthetic operations layer (local)

- Amazon/meta does **not** carry COGS, inventory, or true sales. Generate **seeded synthetic** tables keyed by `product_id` (inventory, suppliers, daily sales, marketing spend, weekly KPIs). See `docs/SYNTHETIC_DATA.md` and `scripts/generate_ops_data.py` (or `scripts/generate_data.py` for agent + ops together).
- **External trends / web:** use a dedicated **market intelligence** agent with search tools; see `docs/AGENT_ARCHITECTURE_TRENDS.md`.

---

## 2) Pipeline Implementation

### 2.1 Stage Tables (raw/staging)

- `stg_home_products_meta`
- `stg_home_reviews`
- `stg_pipeline_runs`

### 2.2 Canonical Tables

- `products`
- `reviews`
- `product_signals`
- `retrieval_corpus`

### 2.3 Pipeline Jobs

1. Ingest raw chunks.
2. Normalize and clean.
3. Build canonical tables.
4. Run quality checks.
5. Emit quality report artifact.

### 2.4 Orchestration

- Keep Prefect flow + CLI entrypoint.
- Keep GitHub Actions scheduled/manual run.
- Log run status and row counts in BigQuery.

---

## 3) Warehouse and Data Model

### 3.1 Required Core Schema

- `products(product_id, title, brand, category, subcategory, price, avg_rating, rating_count, product_document, updated_at)`
- `reviews(review_id, product_id, review_text, review_title, rating, helpful_vote, event_ts)`
- `product_signals(product_id, review_count, positive_ratio, sentiment_score, recency_score, value_score, price_band, trend_features...)`
- `retrieval_corpus(product_id, product_document, category, subcategory, price, avg_rating)`

### 3.2 Optional Tables

- `external_trends` (news/search/social trend signals)
- `decision_logs` (owner accept/reject and rationale)
- `agent_traces` (debug/evaluation of debate outputs)

---

## 4) Feature Engineering and Data Splits

### 4.1 Features

- Text features:
  - sentiment logits/scores
  - complaint themes (keyword/topic aggregates)
- Numeric features:
  - rating aggregates, helpfulness, review velocity
  - price percentile in subcategory
  - value proxies (rating/price, trend-adjusted)
- Retrieval features:
  - dense embeddings for `product_document`

### 4.2 Splits

- Time-aware splits preferred:
  - train: older window
  - val: middle window
  - test: latest window
- If timestamps incomplete: deterministic hash split as fallback.

### 4.3 Why split if not full training?

- Required for fair model comparison.
- Required for reporting validation quality in rubric.
- Used for selecting final production model per agent.

---

## 5) Model Comparison Plan (Required)

### 5.1 Sentiment Agent Comparison

- `VADER` vs `DistilRoBERTa` vs `DeBERTa-v3`
- Metrics:
  - macro F1
  - accuracy
  - confusion matrix
- Pick winner based on quality + runtime.

### 5.2 Pricing/Value Agent Comparison

- `CatBoost` vs `FT-Transformer`
- Metrics:
  - MAE
  - RMSE
  - MAPE (if valid target scale)
- Choose model with best error + stable inference latency.

### 5.3 Retrieval/Comparison Agent

- `all-MiniLM-L6-v2` vs `e5-large-v2` vs `bge-large-en-v1.5`
- Metrics:
  - Recall@K
  - nDCG@K
  - MRR
- Add relevance sanity checks with sampled human validation.

### 5.4 Orchestrator Comparison

- `Phi-3.5-mini` vs `Qwen2.5-7B`
- Metrics:
  - recommendation quality rubric score
  - latency p50/p95
  - agreement with accepted human decisions

---

## 6) Final Model Lock (Post-Comparison)

For each agent, lock:

- primary model
- fallback model
- max latency target
- confidence calibration rule

Expected output:

- `final_model_registry.md` or config block in `models.yaml`.

---

## 7) Multi-Agent Runtime Design

### 7.1 Agents

- Demand/Customer Signal Agent
- Pricing/Margin Agent
- Inventory/Comparison Agent
- Orchestrator

### 7.2 Debate Rounds

- Round 1: independent arguments.
- Round 2: cross-reaction to peer arguments.
- Orchestrator: final synthesis with policy constraints.

### 7.3 Output Contract

Each plan contains:

- `actions[]`
- `impacted_skus[]`
- `expected_impact`
- `risk`
- `confidence`
- `evidence_refs[]`

---

## 8) Policy Layer and Guardrails

Implement deterministic constraints before final output:

- minimum margin floor (proxy or real margin mode)
- max price change cap
- low-confidence action suppression
- must-cite evidence requirement

If policy fails:

- return safer alternative plans.

---

## 9) Web App and Human-in-the-Loop

### 9.1 UI Flow

1. Owner asks business question.
2. App shows context used (category, time window, constraints).
3. App displays 2-3 ranked plans with evidence.
4. Owner accepts/rejects + reason.

### 9.2 Logging

- Persist:
  - query
  - plans shown
  - selected/rejected plan
  - reason
  - latency
  - model versions

---

## 10) Evaluation and Testing

### 10.1 Model-Level Testing

- Run all comparison experiments on locked split.
- Save metrics JSON + charts.

### 10.2 System-Level Testing

- End-to-end scenario tests (goal -> recommendation -> accept/reject log).
- Runtime QoS:
  - latency p50/p95
  - response consistency
  - failure handling

### 10.3 Evidence Artifacts

- confusion matrices
- retrieval quality tables
- error distribution plots
- app screenshots by use case flow

---

## 11) Report and Demo Packaging

### 11.1 Report Mapping

- Ch2/Ch3: data + pipeline + warehouse
- Ch4: model proposals/comparisons/final selections
- Ch5: system architecture + app + APIs
- Ch6: evaluation, visualization, constraints

### 11.2 Demo Script

- Scenario 1: increase revenue
- Scenario 2: cut costs
- Scenario 3: improve margin/reduce dead inventory
- Show one approve and one reject with reason.

### 11.3 Submission Pack

- final report
- slides
- demo video
- appendix A/B/C assets
- organized artifact folder

---

## 12) Execution Milestones

### Milestone A: Data and Pipeline Ready

- subset finalized
- canonical tables built
- quality report passing

### Milestone B: Models Compared

- all comparison runs complete
- metric tables and plots generated
- final model selections locked

### Milestone C: End-to-End App Ready

- debate flow working
- policy layer active
- accept/reject logging complete

### Milestone D: Final Delivery

- report complete
- demo rehearsed
- submission package uploaded

---

## 13) Non-Negotiables

- No random catalog sampling.
- No full-category brute-force ingestion.
- No legacy linear baseline as primary story.
- Keep architecture vertical-configurable (future shoe/apparel pivot via config).
- Every recommendation must be grounded in retrievable evidence.

