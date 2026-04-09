# Sharing the Copilot v2 dataset (Google Drive)

**Snapshot ID (lock this for the team):** `38710839ca6e1009`  
**Domain:** Amazon Reviews 2023 — Home & Kitchen (curated subset).  
**Approx. size (core bundle, no retrieval indexes):** ~**2.4 GB** (mostly `reviews.parquet`).

Teammates should use **the same snapshot ID** everywhere (`--snapshot-id 38710839ca6e1009`, paths below). If you rebuild Stage 1 with different config, the hash changes and everyone must switch together.

---

## 1. What to upload to Google Drive (recommended layout)

Create one top folder on Drive, e.g. **`copilot_v2_snapshot_38710839ca6e1009`**, and **mirror** the repo paths **inside** it so unzip instructions stay simple.

### 1.1 Required — core training / tabular / synthetic (upload these)

| Local path (relative to repo root) | Approx. size | Upload? |
|-------------------------------------|--------------|---------|
| `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/` | ~2.3 GB | **Yes** (entire folder) |
| `copilot-v2/artifacts/splits/38710839ca6e1009/` | ~40 MB | **Yes** (entire folder) |
| `copilot-v2/artifacts/synthetic/38710839ca6e1009/` | ~30 MB | **Yes** (entire folder) |
| `copilot-v2/artifacts/features/38710839ca6e1009/` | ~60 MB | **Yes** (entire folder) |

**Zip tip (optional):** You can zip each of the four folders (or one zip of all four) to avoid Drive partial uploads. Name zips with the snapshot id in the filename.

### 1.2 Optional — retrieval indexes (~1 GB)

Only if teammates must **run retrieval evals or RAG** without rebuilding FAISS from scratch.

| Local path | Approx. size | When to include |
|------------|--------------|-----------------|
| `copilot-v2/artifacts/indexes/38710839ca6e1009/` | ~933 MB | Retrieval agent / end-to-end demo |
| `copilot-v2/artifacts/registry/registry.json` | tiny | Points at winner + paths |
| `copilot-v2/artifacts/evals/38710839ca6e1009/retrieval/` | small | Metrics + tuning JSON (audit trail) |

You may already have a single backup: `copilot-v2/artifacts/registry/retrieval_38710839ca6e1009_bundle_20260402.tar.gz` (~820 MB) — that bundles **indexes + retrieval evals** (not the core `data_snapshots` Parquet).

### 1.3 Do **not** need to upload for “train other agents” (usually)

- Raw JSONL under `copilot-v2/data/raw/` (teammates use **Parquet snapshot** only unless they rebuild Stage 1).
- Python venv, Hugging Face cache (each machine downloads models as needed).
- This markdown file is small; you can upload a copy next to the zips on Drive for convenience.

---

## 2. File-by-file guide (core bundle)

Paths below are under **`copilot-v2/artifacts/`** after unzip into the cloned repo.

### 2.1 `data_snapshots/38710839ca6e1009/`

| File | Role | Typical consumers |
|------|------|-------------------|
| **`products.parquet`** | One row per product: title, brand, category/subcategory, description, price, aggregates, `product_document`. | Tabular features source, joins, display. |
| **`reviews.parquet`** | One row per review: `review_text`, `rating`, timestamps, `product_id`. | **Sentiment** training/eval (canonical split joins). |
| **`product_signals.parquet`** | Per-product aggregates and **enriched** subcategory-relative fields (see `manifest_derived_enrich.json`). | Tabular / pricing / orchestrator context. |
| **`retrieval_corpus.parquet`** | `product_id` + **`product_document`** (structured template after enrich) + facets. | Retrieval indexing / RAG corpus. |
| **`manifest.json`** | Stage 1 provenance: gates, subset caps, **column lists**, row counts, source JSONL paths. | Documentation, schema audit. |
| **`manifest_derived_enrich.json`** | Lists **post–Stage 1** enrichments (structured `product_document`, extra `product_signals` columns). | Know exact columns after enrich. |
| **`reviews_sentiment_balanced.parquet`** | Review table with star-bucket downsampling (less pos skew). | **Optional** sentiment training; document if you use this vs canonical `reviews.parquet`. |
| **`reviews_sentiment_balanced_meta.json`** | Params for balancing run. | Reproducibility. |
| **`priced_subset/`** | Smaller mirror: products with non-null price + aligned `reviews` / `signals` / `retrieval_corpus` + `manifest.json`. | Teammates who only need priced SKUs (e.g. some pricing experiments). |

**Join keys:** `product_id` (string) everywhere; `review_id` + `product_id` for reviews.

### 2.2 `splits/38710839ca6e1009/`

| File | Role | Typical consumers |
|------|------|-------------------|
| **`products_split.parquet`** | `product_id` → `split` ∈ {`train`, `val`, `test`}. | Any model keyed by product. |
| **`reviews_split.parquet`** | `review_id`, `product_id` → `split`. | **Sentiment** (must join to reviews). |
| **`split_config.json`** | Time/leakage notes, seed, fractions (e.g. 70/15/15). | Paper / advisor; defines how splits were built. |

**Usage:** Inner-join split tables to the base tables **before** training or selecting eval rows. **Do not tune on test.**

### 2.3 `synthetic/38710839ca6e1009/`

| File | Role | Typical consumers |
|------|------|-------------------|
| **`inventory_skus.parquet`** | Synthetic inventory, margin, velocity class, stock, `product_id`. | **Pricing / ops** agent, tabular joins. |
| **`sales_daily.parquet`** | Long-format daily sales per `product_id`. | Demand features, pricing labels pipeline. |
| **`synthetic_ops_meta.json`** | Generator seed, horizon, scale. | Reproducibility. |

Join to products / tabular features on **`product_id`**.

### 2.4 `features/38710839ca6e1009/`

| File | Role | Typical consumers |
|------|------|-------------------|
| **`tabular_features.parquet`** | Wide table: product attributes + signals + inventory + sales aggregates + **`split`**. | **Pricing** (CatBoost / FT-Transformer / TabPFN), baselines. |
| **`tabular_features_meta.json`** | Column list, target candidates, build notes. | Feature audit. |

**Label note:** `recommended_price_change_pct` may be added in a later pipeline step; until then use targets documented in meta / `AGENT_PLANS.md`.

---

## 3. Teammate setup (after download from Drive)

1. **Clone** this repo (same branch / commit as the team agrees, if relevant).
2. **Restore folders** so these exist relative to repo root (same paths as above):

   - `copilot-v2/artifacts/data_snapshots/38710839ca6e1009/`
   - `copilot-v2/artifacts/splits/38710839ca6e1009/`
   - `copilot-v2/artifacts/synthetic/38710839ca6e1009/`
   - `copilot-v2/artifacts/features/38710839ca6e1009/`

3. In scripts/configs, set:

   - **`snapshot_id`:** `38710839ca6e1009`
   - **`artifacts_root`:** `copilot-v2/artifacts` (default in many configs)

4. **Python:** `pip install -r copilot-v2/requirements.txt` (or project venv).

5. **Read next:** `Docs/DATA_FLOW_AND_AGENT_PROTOCOL.md`, `Docs/AGENT_PLANS.md`, `copilot-v2/docs/EXPERIMENT_RECORDING.md` (in-repo copies under `Docs/` / `copilot-v2/docs/` as available).

---

## 4. Agent → files quick map

| Agent | Primary files |
|-------|----------------|
| **Sentiment** | `reviews.parquet` ⋈ `reviews_split.parquet`; optional `reviews_sentiment_balanced.parquet` |
| **Pricing / tabular** | `tabular_features.parquet`; joins to `synthetic/` as needed |
| **Retrieval** | `retrieval_corpus.parquet`, `products.parquet` (titles as queries); **indexes** optional if uploaded |
| **Orchestrator / policy** | Uses outputs of above + `inventory_skus.parquet` for constraints |

---

## 5. License and sharing

- Underlying data is derived from the **Amazon Reviews 2023** corpus (McAuley/Lab). Use only per the **dataset license** and your course rules.
- This bundle is **for team / class use**; avoid public redistribution of processed dumps unless allowed.
- **`manifest.json`** records source paths on the builder’s machine; teammates do not need those JSONL files if they only consume Parquet.

---

## 6. Checklist before you click “Share” on Drive

- [ ] All four **required** directories uploaded (or one zip per directory with clear names).
- [ ] Folder or zip names include **`38710839ca6e1009`**.
- [ ] Optional: retrieval **indexes** or **`retrieval_*_bundle_*.tar.gz`** for teammates who need prebuilt retrieval.
- [ ] Share link permissions set (e.g. team Google accounts only).
- [ ] Paste this doc (or a PDF export) into the same Drive folder so usage is obvious.
