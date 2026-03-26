# Data snapshot (locked baseline)

One-page summary for report Ch2/Ch3 and reproducibility. Aligns with `artifacts/stage1/curation_summary.json` and `data/agent_dataset/agent_dataset_manifest.json`.

## Row counts

| Table / artifact | Rows | Notes |
|------------------|------|--------|
| `products.parquet` | 1,500 | Agent dataset |
| `reviews.parquet` | 88,837 | Agent dataset |
| `product_signals.parquet` | 1,500 | Agent dataset |
| `retrieval_corpus.parquet` | 1,500 | Agent dataset |

**Curation pools (replicated baseline mode):** review text pool *n* = 22,067; meta title pool *n* = 14,994 (see `curation_summary.json` → `quality_report.pools`).

## Synthetic ops data (`data/synthetic/`)

Generated alongside the agent tables; see `data/synthetic/synthetic_store_manifest.json` for file list and parameters. Default `generate_data.py` uses `--seed 42` and `SyntheticStoreConfig(random_seed=49)` (i.e. `seed + 7`).

## Regeneration

From repo root (after raw pools exist under `data/raw/` as in the manifest):

```text
python seller-copilot/scripts/generate_data.py --quick --seed 42
```

## Environment fingerprint

| Item | Value |
|------|--------|
| `requirements.txt` SHA-256 | `a29fe1ff61399cfbc35f3579ed5c4b0f1de540c81452b56b4b358eb4eb9666f1` |
| Intended Python | 3.10–3.12 (project stack; avoid bleeding-edge interpreters until wheels catch up) |
| Default data RNG seed | `42` (`generate_data.py --seed`) |

## Validation

```text
python seller-copilot/scripts/validate_agent_data.py
```

Use `--strict` to fail on warnings.

## BigQuery

If the rubric requires a warehouse upload, see `docs/GCP_SETUP.md` and `python seller-copilot/scripts/upload_to_bigquery.py --reset`.
