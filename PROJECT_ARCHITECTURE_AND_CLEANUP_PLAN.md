# CoD Multi-Agent E-Commerce: Architecture + Cleanup Plan

Date: 2026-02-21  
Scope: repository cleanup + production-ready architecture plan for Human-AI collaborative agents (BigQuery + Prefect + Jupyter cloud-first)

## Project Constraint (Mandatory)

This project must run at zero paid cost:

- No paid API usage (no token billing).
- No subscriptions or credits.
- Use only open-source models/tools and free tiers.
- Default execution must be cloud-first.
- Local GPU use is allowed only for training/agent setup experiments when cloud free runtimes are insufficient.

## 1) What the Professor/Rubrics Require (from `text`)

The grading emphasis is not only on models, but full system quality:

- Propose and compare multiple models (3+ for good, 5+ for excellent).
- Justify model choices and show strengths/limits.
- Define evaluation methods and show actual results.
- Define system boundary, actors, use cases.
- Design complete architecture (AI components + cloud integration + data management + UI/visualization).
- Deliver intelligent solution with clear dataset inputs, outputs, and APIs.
- Show system support environment (tools/platforms/frameworks).
- Provide testing evidence, performance/quality evaluation, and demo-ready web portal.

Implication: a strong submission needs both ML rigor and software architecture discipline.

## 2) Current Repo Audit (What exists vs gaps)

Observed strengths:

- Good exploratory notebook flow exists (`01` to `07` pipeline notebooks).
- Evidence of ingestion/transformation/warehouse/embedding/orchestration work.
- Streamlit app prototype already wires retrieval + ranking + pricing flow.
- Processed parquet assets exist under `data/processed/`.

Critical issues to fix:

- `notebooks/agents/*.py` files are empty (no modular agent implementation).
- `notebooks/app.py` expects `data/products_50k.csv` and `data/faiss_index_50k.bin`, but these are currently missing in repo.
- `data/processed/amazon_reviews_clean.csv` appears to contain only headers (no rows).
- Pipeline logic lives mostly in notebooks; low reproducibility for CI/CD.
- Hard-coded cloud context and historical execution artifacts inside notebooks.
- Secret/token leakage risk: `notebooks/env` contains a token-like value and command.
- `.gitignore` ignores all `*.csv` and `*.parquet`; this can hide whether required demo artifacts are reproducible from code.

## 3) Target Architecture (BigQuery + Prefect + Jupyter, cloud-first)

### High-level design

1. Data Ingestion Layer
- Source datasets pulled by version-pinned scripts.
- Raw snapshots persisted in cloud artifact storage (for reproducibility) and loaded into BigQuery staging tables.

2. Data Processing Layer
- BigQuery SQL transformations + Python preprocessing jobs orchestrated by Prefect.
- Outputs written to BigQuery curated datasets and exported as parquet artifacts for reproducibility.

3. Feature + Retrieval Layer
- Product/review embeddings generated in batch.
- Store retrieval metadata in BigQuery tables; persist vector index artifacts as files.
- Hybrid retrieval: vector similarity + metadata filters + optional lexical fallback.

4. Agent Layer (modular services)
- Discovery Agent (candidate retrieval)
- Sentiment Agent (user voice synthesis)
- Ranking Agent (relevance rerank)
- Pricing/Value Agent (value-for-money analysis)
- Orchestrator Agent (debate rounds + final synthesis)

5. App/API Layer
- Web app UI consumes backend API.
- Backend endpoints expose `query -> recommendation + rationale + agent traces`.
- Serving APIs read curated outputs from BigQuery and model artifacts from cloud storage.

6. Observability + Evaluation Layer
- Store prompts, versions, latency, outputs, and eval metrics in BigQuery evaluation tables.
- Track pipeline runs and retries in Prefect run history.

### Suggested repository structure

```text
cod-multiagent-ecommerce/
  apps/
    web/                      # frontend
    api/                      # orchestration + endpoints
  src/
    agents/
      discovery.py
      sentiment.py
      ranking.py
      pricing.py
      orchestrator.py
    pipelines/
      ingest.py
      transform.py
      embed.py
      evaluate.py
    schemas/
      contracts.py            # pydantic I/O contracts
    services/
      bigquery_client.py
      model_router.py
  flows/
    prefect_flow.py           # orchestrated pipeline flow
  infra/
    bigquery/
      schema.sql
      transforms.sql
  data/
    sample/                   # tiny test fixtures only
  tests/
    unit/
    integration/
    eval/
```

## 4) Dataset Strategy (Solid and defensible)

Use the four datasets already identified in project scope:

- Twitter Customer Support: intent + sentiment signal for support interactions.
- Online Retail II: transactional behavior for recommendation and segmentation.
- Telco Customer Churn: churn-style predictive patterns and retention signals.
- Amazon Reviews 2023: product/review text for retrieval, sentiment, and value analysis.

Rules for a "solid dataset" deliverable:

- Create dataset cards per source (license, schema, row count, time range, known biases).
- Define one canonical schema for each entity: `users`, `products`, `events`, `reviews`, `sessions`.
- Add strict dedupe policy and leakage checks before train/val/test split.
- Use temporal split where applicable (avoid future leakage in recommendation/churn tasks).
- Version every dataset artifact with hash + pipeline run id.

### Current dataset readiness verdict (local repo check, 2026-02-21)

Verdict: partially ready.

- Good enough for prototype agents: yes (Amazon-focused discovery/ranking/pricing + basic sentiment).
- Good enough for final rubric-heavy multi-agent system: not yet.

Why not yet:

- The repo currently shows mostly Amazon processed assets under `data/processed/`.
- `data/processed/amazon_reviews_clean.csv` appears header-only (no rows).
- Required cross-domain datasets in project scope (Twitter support, Online Retail II, Telco churn) are not yet integrated here.
- Gold evaluation labels are missing for robust agent benchmarking (relevance labels, aspect sentiment labels, orchestrator success labels).

To make datasets final-ready:

1. Add all planned datasets into one canonical schema (`users/products/events/reviews/sessions`).
2. Build evaluation sets for retrieval/ranking/sentiment/orchestration.
3. Enforce temporal split and leakage checks in pipeline code.
4. Add automated data-quality checks (nulls, duplicates, schema drift, class imbalance).
5. Version every dataset and model-input artifact with run id + hash.

## 5) Preprocess + Data Pipeline Plan

Pipeline stages:

1. Ingest: validate schema, types, null thresholds, and duplicate keys.
2. Clean: normalize text, prices, timestamps, and category taxonomies.
3. Feature: derive sentiment labels/aspects, product aggregates, price bands, behavior features.
4. Embed: generate vectors for product text + review summaries.
5. Index: build vector index and retrieval tables.
6. Evaluate: run regression metrics + retrieval metrics + agent-level quality checks.
7. Publish: materialize serving tables/views for app.

Data contracts to enforce between stages:

- `product_document` contract (id, title, description, category, price, embedding_version)
- `review_signal` contract (item_id, sentiment_score, aspect_scores, confidence)
- `retrieval_candidate` contract (query_id, item_id, score_source, score, timestamp)

## 6) Model Stack Recommendation (Free-only)

As-of 2026-02-21, recommended open-source stack:

Tier A (best quality under free constraint):

- Orchestrator + debate synthesis: `Qwen2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`.
- Embeddings: `BAAI/bge-large-en-v1.5`.
- Reranking: `BAAI/bge-reranker-v2-m3`.
- Sentiment/aspect baseline: `cardiffnlp/twitter-roberta-base-sentiment-latest`.

Tier B (lighter local compute):

- Orchestrator/ranking: `Qwen2.5-3B-Instruct` or `Mistral-7B-Instruct`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Rerank fallback: cross-encoder open reranker or cosine similarity only.

Tier C (CPU-first emergency fallback):

- Orchestrator: distilled small instruct model via `llama.cpp`/`Ollama`.
- Retrieval: FAISS + lexical BM25 fallback.
- Sentiment: scikit-learn TF-IDF + linear classifier baseline.
- Keep the same evaluation harness so swaps remain measurable.

Important: do not claim "best" by name alone. Run your own benchmark harness:

- Retrieval: Recall@K, nDCG@K, MRR.
- Ranking: nDCG, pairwise accuracy.
- Sentiment/aspect: macro F1, calibration/confidence.
- End-to-end agent system: success@1, explanation quality rubric, latency p95, cost/request (should be near zero).

Runtime/deployment (free):

- Cloud notebooks (Jupyter/Colab/Kaggle) for training and batch jobs.
- Pipeline orchestration: Prefect (free tier/open-source) triggering BigQuery and notebook tasks.
- Vector search artifacts: `FAISS` index files stored in cloud artifacts.
- Optional local GPU runs only for training/agent setup acceleration.

## 7) Agent Architecture (Human-collaborative CoD)

Recommended debate protocol:

1. Discovery proposes top-K candidates.
2. Sentiment/Ranking/Pricing agents each produce:
- claim
- evidence
- confidence
- recommended product(s)
3. Orchestrator runs conflict resolution round:
- identify disagreements
- request one rebuttal from each debating agent
4. Final synthesis:
- winner + runner-up
- why selected
- uncertainty and fallback suggestion
- "human override" controls in UI

Human-collaboration requirement:

- User can inspect each agent's reasoning trace.
- User can adjust preferences (budget, brand, quality priority) and rerun.
- User can accept/reject recommendation and feed explicit feedback.

## 8) Web App Plan (Rubric-aligned)

Minimum app sections:

- Query/assistant page (core recommendation flow).
- Agent debate panel (trace + confidence + evidence snippets).
- Evaluation dashboard (offline metrics + runtime latency/cost).
- Data lineage panel (dataset versions and model versions used).

If time is tight:

- Keep Streamlit for final demo but modularize backend first.
- If time allows, upgrade to React/Next.js frontend with FastAPI backend for cleaner architecture story.

## 9) Cloud Stack Design Recommendations (BigQuery + Prefect + Jupyter)

Core components:

- BigQuery Sandbox for staging, transformation, analytics, and evaluation tables.
- Prefect for scheduled orchestration, retries, and run tracking.
- Jupyter notebooks (cloud runtime) for model training, embedding generation, and analysis.
- Cloud artifact storage for parquet snapshots and FAISS/model artifacts.

Data model sketch:

- `products`, `reviews`, `customers`, `sessions`, `queries`, `recommendations`, `agent_runs`, `evaluation_runs`.
- Retrieval artifacts managed as versioned files plus metadata tables.
- Materialized views/tables for dashboard metrics.

Operational constraints and mitigations:

- BigQuery Sandbox limits and table expiration require scheduled refresh/rebuild jobs.
- Keep canonical ingestion scripts and snapshots so environment can be recreated quickly.
- Keep dataset size demo-focused and rubric-aligned.

Security and ops:

- Move all secrets to env vars (never in repo).
- Use service-account credentials only in pipeline/runtime environments.
- Keep schema as SQL files under version control (`infra/bigquery/*.sql`).
- Add CI job: lint + tests + pipeline smoke test.

## 10) Priority Cleanup Backlog (Execution Order)

P0 (immediate):

1. Remove secret/token artifacts (`notebooks/env`) and rotate compromised credentials.
2. Convert current monolithic `notebooks/app.py` logic into `src/agents/*.py` modules.
3. Make data artifact generation reproducible via scripts (not notebook-only).
4. Fix broken runtime assumptions (generate or load required `products_50k` + index from pipeline).

P1 (core architecture):

1. Set up BigQuery dataset/schema, Prefect flow definitions, and seed jobs.
2. Implement API layer with strict request/response contracts and cloud-model adapters.
3. Add offline evaluation harness and baseline metrics table.

P2 (demo polish):

1. Add debate UI + feedback capture.
2. Add observability dashboards (latency/cost/quality).
3. Add system test scenarios mapped to rubric use cases.

## 11) Rubric Coverage Map (what to show in final demo/doc)

- Model proposals/comparison: benchmark table across at least 5 model combinations.
- Model support environment: architecture + dataflow + deployment diagram.
- Evaluation: offline metrics + online demo traces + latency/cost.
- System requirements/design: actors, use cases, boundary diagram.
- Intelligent solution: live API + end-to-end recommendation evidence.
- Web portal: clear UI flow, visualizations, and tested use cases.

## 12) Suggested Next Build Sprint (7 days)

Day 1-2:
- repo cleanup + secrets fix + module scaffolding + BigQuery schema + Prefect flow setup

Day 3-4:
- ingestion/transform/embed scripts + candidate retrieval API

Day 5:
- sentiment/ranking/pricing agent APIs + orchestrator protocol

Day 6:
- evaluation harness + dashboard metrics tables

Day 7:
- web integration + demo scripts + rubric checklist validation

---

## References used for current recommendations

- BigQuery Sandbox docs: https://cloud.google.com/bigquery/docs/sandbox
- BigQuery quotas and limits docs: https://cloud.google.com/bigquery/quotas
- Prefect docs: https://docs.prefect.io/
- Jupyter docs: https://docs.jupyter.org/en/latest/
- Hugging Face Datasets docs: https://huggingface.co/docs/datasets/index
- Kaggle notebooks docs: https://www.kaggle.com/docs/notebooks
- Google Colab docs: https://research.google.com/colaboratory/faq.html
- Sentence Transformers docs: https://www.sbert.net/
- FAISS docs: https://github.com/facebookresearch/faiss
- Hugging Face model card (`BAAI/bge-large-en-v1.5`): https://huggingface.co/BAAI/bge-large-en-v1.5
- Hugging Face model card (`BAAI/bge-reranker-v2-m3`): https://huggingface.co/BAAI/bge-reranker-v2-m3
- Hugging Face model card (`cardiffnlp/twitter-roberta-base-sentiment-latest`): https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
