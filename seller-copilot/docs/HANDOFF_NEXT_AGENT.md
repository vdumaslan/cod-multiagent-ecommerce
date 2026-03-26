# Handoff: full context for the next agent (submission + data + what’s next)

Use this document when continuing work in a **new chat**. It replaces relying on an old skill file alone.

---

## 1. What this project is (submission track)

**Seller Copilot** — a **business decision copilot** for e‑commerce: owner states a goal; the system returns **2–3 ranked action plans** with **evidence and risk**; **human approve/reject** with optional reason; feedback for re-ranking.

**Locked vertical for this submission:** **Amazon Reviews 2023 — Home & Kitchen** (`McAuley-Lab/Amazon-Reviews-2023`), not Costco/Unwrangle for this phase.

**Core docs (source of truth for rubric):**

| Doc | Purpose |
|-----|---------|
| [End_to_End_Implementation_Plan.md](End_to_End_Implementation_Plan.md) | Deliverables, milestones A–D, model comparison plan (§5), Streamlit, policy, report structure |
| [01_constraints_and_success.md](01_constraints_and_success.md) | Zero paid APIs, BigQuery Sandbox, Prefect, FAISS, Streamlit, **≥4 distinct models**, debate architecture |
| [SUBMISSION.md](SUBMISSION.md) | Short checklist: validate → optional BQ → models → demo → report |
| [GCP_SETUP.md](GCP_SETUP.md) | BigQuery credentials |
| [AGENT_ARCHITECTURE_TRENDS.md](AGENT_ARCHITECTURE_TRENDS.md) | Optional “market intelligence” agent + web search pattern |

**Final deliverables (from implementation plan §0.3):**

- Working **Streamlit** demo  
- **Reproducible** data pipeline + artifacts  
- **Model comparison** results + **final model justifications**  
- **Final report** + appendix evidence pack  

---

## 2. Current data state (what exists on disk)

The user ran **`python seller-copilot/scripts/generate_data.py --quick`** successfully. This produces **agent-ready Parquet** + **ops/synthetic Parquet** in one run (no separate `generate_ops_data.py` needed unless refreshing ops only).

**Approximate row counts (locked baseline):**

| Artifact | Rows |
|----------|------|
| `data/agent_dataset/products.parquet` | 1,500 |
| `data/agent_dataset/reviews.parquet` | ~88,837 |
| `data/agent_dataset/product_signals.parquet` | 1,500 |
| `data/agent_dataset/retrieval_corpus.parquet` | 1,500 |
| `data/agent_dataset/agent_dataset_manifest.json` | metadata + columns |

**Pools used for text (from local raw JSONL):** ~22k review texts, ~15k meta titles (see `DATA_SNAPSHOT.md` / `curation_summary.json`).

**Synthetic ops** under `data/synthetic/`: `inventory_skus`, `suppliers`, `product_supplier_map`, `sales_daily`, `marketing_spend_daily`, `store_kpis_weekly`, `synthetic_store_manifest.json`.

**Duplicates for debugging:** `seller-copilot/artifacts/stage1/*_curated.parquet`, `curation_summary.json`, `quality_report.json`.

**One-page snapshot for report Ch2/Ch3:** [DATA_SNAPSHOT.md](DATA_SNAPSHOT.md).

**Scripts:**

| Script | Role |
|--------|------|
| `scripts/generate_data.py` | **`--quick`** = fast preset (1500 products, smaller pools). Full: `--products`, `--pool-reviews`, `--pool-meta`. |
| `scripts/generate_ops_data.py` | Ops only; needs `products.parquet` already. |
| `scripts/validate_agent_data.py` | Validates files + joins + price stats. |
| `scripts/upload_to_bigquery.py` | Loads agent + synthetic Parquet to BQ (see `GCP_SETUP.md`). |
| `src/data_acquisition/scripts/run_stage1.py` | **Optional** full two-pass scan over **43 GB** raw JSONL (hours)—not required if using `generate_data.py`. |

**Raw data (optional, large):** `data/raw/amazon_reviews_2023/*.jsonl` (gitignored).

---

## 3. Environment note (Python / NumPy)

Use **Python 3.10–3.12** in a **venv** (e.g. `.venv`). **Avoid Python 3.14+** for this stack until NumPy/PyArrow wheels are stable—validation and Parquet may fail with DLL errors.

**Validate:** from repo root, `python seller-copilot/scripts/validate_agent_data.py`.

---

## 4. What the submission execution plan said (next work, in order)

**Phase A — Data locked** (mostly done)

- Run `validate_agent_data.py` on the user’s machine (3.10–3.12).  
- `DATA_SNAPSHOT.md` exists.  
- Optional: **BigQuery** upload if rubric requires warehouse (`upload_to_bigquery.py`).  

**Phase B — Splits + features** (not done)

- Reproducible **train/val/test** (time split on `reviews.event_ts` preferred, else hash split with fixed seed).  
- Join `products` + `product_signals` (+ optional synthetic) for tabular models.  

**Phase C — Model comparisons** (not done as scripted pipeline under `artifacts/evals/`)

- **Sentiment:** VADER vs DistilRoBERTa (vs optional DeBERTa); macro F1, confusion matrices; **proxy labels** from `rating` or small hand labels.  
- **Tabular:** CatBoost vs **FT-Transformer** or second model (time-boxed).  
- **Retrieval:** two sentence-transformers + **FAISS**; Recall@K, nDCG, MRR.  
- **Orchestrator:** local small LLM or **stub** with debate-shaped JSON (no paid APIs per constraints).  

**Phase D — System**

- **Debate** architecture (≥2 rounds), **policy** guardrails, **evidence refs**.  
- **Streamlit** app: goal → 2–3 plans → approve/reject + **logging** (query, plans, choice, reason, latency).  
- **Prefect:** Stage 1 flow exists; optional second flow for “eval pipeline” if rubric asks.  

**Phase E — Packaging**

- Screenshots, `final_model_registry.md`, report chapters, demo video, zip.  

---

## 5. Conflict: `.cursor/skills/seller-copilot-pivot/SKILL.md` vs repo reality

There is a **Cursor skill** `seller-copilot-pivot` that says:

- **No Amazon datasets**; use **Costco via Unwrangle** instead  
- Minimize LLM usage; different file paths (`pipeline.yaml`, `ingest_sources.py`) that **may not exist** in the current tree  

**The actual codebase and submission path** use:

- **McAuley Amazon Reviews 2023** Home & Kitchen  
- `scripts/generate_data.py`, `src/data_acquisition/`, etc.  

**For the course submission, treat the implementation plan + this repo as canonical.** Do **not** rip out Amazon data or switch to Costco unless the **user explicitly** pivots and updates docs/manifests. If a skill is applied, **ignore pivot rules** unless the user confirms a pivot.

---

## 6. Quick checklist for the next agent

1. Confirm Python **3.10–3.12** venv; run `validate_agent_data.py`.  
2. If rubric requires cloud: run `upload_to_bigquery.py` + `.env` per `GCP_SETUP.md`.  
3. Implement **Phase B** (split script + notebook).  
4. Implement **Phase C** (eval scripts → `artifacts/evals/`, FAISS under `artifacts/faiss/`).  
5. Add **`final_model_registry.md`**.  
6. Build **Streamlit** + **decision logs** + **policy** (Phase D).  
7. **Report + screenshots + zip** (Phase E).  

---

## 7. Key file paths (repo root = `cod-multiagent-ecommerce/`)

- Data: `seller-copilot/data/agent_dataset/`, `seller-copilot/data/synthetic/`  
- Artifacts: `seller-copilot/artifacts/stage1/`  
- Flow: `seller-copilot/src/data_acquisition/flows/stage1_amazon_hk_flow.py`  
- Quality: `seller-copilot/src/data_acquisition/quality.py`  
- Synthetic agent builder: `seller-copilot/src/synthetic_agent_dataset/build.py`  
- Ops generator: `seller-copilot/src/synthetic_store/generator.py`  

---

*Last aligned with: submission execution plan + current `generate_data.py --quick` baseline.*
