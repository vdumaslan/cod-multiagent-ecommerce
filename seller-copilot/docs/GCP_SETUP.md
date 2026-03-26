# GCP / BigQuery credentials (Seller Copilot)

Use this when loading curated data to BigQuery (`upload_to_bigquery.py` or Stage 1 with `require_bigquery: true`).

## 1. Create a service account (Google Cloud Console)

1. Open [Google Cloud Console](https://console.cloud.google.com/) → your project (or create one).
2. **IAM & Admin** → **Service Accounts** → **Create service account**.
3. Grant roles (minimum for this pipeline):
   - **BigQuery Data Editor**
   - **BigQuery Job User**
4. **Keys** → **Add key** → **JSON** → download the file.

Keep this JSON **private** — do not commit it.

## 2. Where to put credentials locally

**Option A — file path (recommended for laptops)**

1. Save the JSON somewhere outside the repo, e.g.  
   `C:\secrets\gcp-seller-copilot-sa.json`
2. Create or edit **`seller-copilot/.env`** (gitignored):

```env
GCP_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET=seller_copilot_prod
BIGQUERY_LOCATION=US
GOOGLE_APPLICATION_CREDENTIALS=C:/secrets/gcp-seller-copilot-sa.json
```

Use **forward slashes** in paths on Windows to avoid escaping issues.

**Option B — JSON in env (CI / ephemeral)**

Paste the **entire** JSON as one line into **`GCP_SA_KEY_JSON`** in `seller-copilot/.env`:

```env
GCP_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET=seller_copilot_prod
BIGQUERY_LOCATION=US
GCP_SA_KEY_JSON={"type":"service_account",...}
```

`run_stage1.py` and `upload_to_bigquery.py` will write `seller-copilot/.secrets/gcp-sa-from-env.json` and set `GOOGLE_APPLICATION_CREDENTIALS` automatically.

## 3. Commands after `.env` is set

From **repository root** (`cod-multiagent-ecommerce/`):

```powershell
cd seller-copilot
..\.venv\Scripts\python scripts\upload_to_bigquery.py --dry-run
..\.venv\Scripts\python scripts\upload_to_bigquery.py --reset
```

Or load everything in one go from Stage 1 + BigQuery:

```powershell
$env:STAGE1_REQUIRE_BIGQUERY = "true"
..\.venv\Scripts\python src\data_acquisition\scripts\run_stage1.py --config config\stage1_amazon_hk_local.yaml
```

(Requires local raw JSONL paths in that config and `use_local_files: true`.)

## 4. What gets created

Dataset: `BIGQUERY_DATASET` (default `seller_copilot_prod`), location `BIGQUERY_LOCATION`.

**Agent tables:** `products`, `reviews`, `product_signals`, `retrieval_corpus`  
**Synthetic tables:** `inventory_skus`, `suppliers`, `product_supplier_map`, `sales_daily`, `marketing_spend_daily`, `store_kpis_weekly`

## 5. Troubleshooting

| Issue | What to check |
|--------|----------------|
| `DefaultCredentialsError` | `GOOGLE_APPLICATION_CREDENTIALS` points to a real file, or `GCP_SA_KEY_JSON` is set. |
| `403 Access Denied` | Service account has BigQuery roles on the project. |
| `404 Dataset` | Run once with `upload_to_bigquery.py` — it calls `ensure_dataset`. |
| Wrong project | `GCP_PROJECT_ID` matches the project where the dataset lives. |
