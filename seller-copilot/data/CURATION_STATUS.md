# Data curation status

## Generate data (quick)

```bash
python seller-copilot/scripts/generate_data.py --quick
```

## Full Stage 1 (local, full-file scan)

A **full** run scans ~43 GB of raw JSONL twice (reviews) + meta once. **Expect many hours.**

Command (from repo root):

```powershell
$env:STAGE1_REQUIRE_BIGQUERY = "false"
python seller-copilot/src/data_acquisition/scripts/run_stage1.py --config seller-copilot/config/stage1_local_agent.yaml
```

When it finishes, you should see:

- `data/agent_dataset/products.parquet`, `reviews.parquet`, `product_signals.parquet`, `retrieval_corpus.parquet`, `agent_dataset_manifest.json`

## Synthetic store data (after Stage 1)

Regenerate so `product_id` sets match:

```bash
python seller-copilot/scripts/generate_ops_data.py
```

## Validate

```bash
python seller-copilot/scripts/validate_agent_data.py
python seller-copilot/scripts/validate_agent_data.py --strict
```

## Upload to BigQuery

See **`docs/GCP_SETUP.md`**, then:

```bash
python seller-copilot/scripts/upload_to_bigquery.py --dry-run
python seller-copilot/scripts/upload_to_bigquery.py --reset
```
