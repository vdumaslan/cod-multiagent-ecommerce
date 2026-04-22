"""Build full-coverage pricing feature table from raw source parquets.

Joins products + product_signals + inventory_skus + sales_daily into a single
feature table matching the schema TabPFN was trained on, at full product coverage.

Sources:
  artifacts/data_snapshots/{snapshot_id}/products.parquet
  artifacts/data_snapshots/{snapshot_id}/product_signals.parquet
  artifacts/synthetic/{snapshot_id}/inventory_skus.parquet
  artifacts/synthetic/{snapshot_id}/sales_daily.parquet

Output:
  artifacts/features/{snapshot_id}/pricing_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"


def run(
    snapshot_id: str = SNAPSHOT_ID,
    artifacts_root: Path | None = None,
) -> Path:
    root = artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
    snap = root / "data_snapshots" / snapshot_id
    syn = root / "synthetic" / snapshot_id
    out_path = root / "features" / snapshot_id / "pricing_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading products.parquet...")
    products = pd.read_parquet(
        snap / "products.parquet",
        columns=[
            "product_id", "price", "avg_rating", "rating_count", "review_count",
            "review_count_agg", "avg_review_rating", "avg_helpful_vote",
        ],
    )

    print("Loading product_signals.parquet...")
    signals = pd.read_parquet(
        snap / "product_signals.parquet",
        columns=[
            "product_id", "positive_ratio", "avg_star_rating", "recent_review_ratio_90d",
            "days_since_last_review", "price_percentile_in_subcategory",
            "rating_vs_subcategory_mean", "subcategory_median_price", "subcategory_mean_price",
            "subcategory_mean_rating", "subcategory_product_count", "inferred_pack_units",
            "price_per_unit", "subcategory_mean_price_per_unit",
        ],
    )

    print("Loading inventory_skus.parquet...")
    skus = pd.read_parquet(
        syn / "inventory_skus.parquet",
        columns=["product_id", "on_hand_units", "safety_stock_units"],
    )

    print("Loading sales_daily.parquet...")
    sales = pd.read_parquet(
        syn / "sales_daily.parquet",
        columns=["product_id", "gross_revenue_usd", "units_sold", "return_units"],
    )
    sales_agg = sales.groupby("product_id").agg(
        mean_daily_revenue=("gross_revenue_usd", "mean"),
        total_units_sold=("units_sold", "sum"),
        total_returns=("return_units", "sum"),
        n_sale_days=("gross_revenue_usd", "count"),
    ).reset_index()

    print("Joining...")
    df = (
        products
        .merge(signals, on="product_id", how="left")
        .merge(skus, on="product_id", how="left")
        .merge(sales_agg, on="product_id", how="left")
    )

    df["log_price"] = np.log1p(df["price"].fillna(0))
    df["rating_price_ratio"] = df["avg_rating"].fillna(0) / df["price"].replace(0, np.nan).fillna(1)

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows, {len(df.columns)} columns → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default=SNAPSHOT_ID)
    args = parser.parse_args()
    run(snapshot_id=args.snapshot_id)
