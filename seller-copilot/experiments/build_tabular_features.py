#!/usr/bin/env python3
"""
Join products + product_signals + synthetic aggregates for tabular (price / margin) models.

Requires artifacts/splits/products_split.parquet from build_splits.py.

Writes: artifacts/splits/tabular_features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_AGENT, DATA_SYN, SPLITS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, default=DATA_AGENT)
    parser.add_argument("--synthetic-dir", type=Path, default=DATA_SYN)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    agent = args.agent_dir.resolve()
    syn = args.synthetic_dir.resolve()
    sp = args.splits_dir.resolve()

    prod = pd.read_parquet(agent / "products.parquet")
    sig = pd.read_parquet(agent / "product_signals.parquet").drop(
        columns=["review_count"], errors="ignore"
    )
    psplit = pd.read_parquet(sp / "products_split.parquet")
    inv = pd.read_parquet(syn / "inventory_skus.parquet")
    sales = pd.read_parquet(syn / "sales_daily.parquet")

    m = prod.merge(sig, on="product_id", how="left", suffixes=("", "_sig"))
    m = m.merge(inv, on="product_id", how="left")
    m = m.merge(psplit, on="product_id", how="left")

    agg = (
        sales.groupby("product_id", as_index=False)
        .agg(
            mean_daily_revenue=("gross_revenue_usd", "mean"),
            total_units_sold=("units_sold", "sum"),
            total_returns=("return_units", "sum"),
            n_sale_days=("sale_date", "count"),
        )
    )
    m = m.merge(agg, on="product_id", how="left")

    for c in ["mean_daily_revenue", "total_units_sold", "total_returns", "n_sale_days"]:
        if c in m.columns:
            m[c] = m[c].fillna(0.0)

    # Simple derived features
    m["log_price"] = np.log1p(pd.to_numeric(m["price"], errors="coerce").clip(lower=0))
    m["rating_price_ratio"] = pd.to_numeric(m["avg_rating"], errors="coerce") / (
        pd.to_numeric(m["price"], errors="coerce").replace(0, np.nan)
    )

    out = sp / "tabular_features.parquet"
    m.to_parquet(out, index=False)
    meta = {
        "n_rows": len(m),
        "columns": list(m.columns),
        "target_candidates": ["price", "margin_pct", "log_price"],
        "categorical": [c for c in ["brand", "category", "subcategory"] if c in m.columns],
    }
    (sp / "tabular_features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
