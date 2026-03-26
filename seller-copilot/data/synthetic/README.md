# Synthetic store data (generated)

These files are **not** from Amazon. They mimic **inventory, COGS, sales, and marketing** so agents and models have realistic operational context.

**Generate** (after `agent_dataset/products.parquet` exists):

```bash
python seller-copilot/scripts/generate_ops_data.py
```

See `docs/SYNTHETIC_DATA.md` for schemas. Parquet blobs are gitignored.
