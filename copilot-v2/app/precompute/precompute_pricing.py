"""Batch pricing inference over tabular_features → pricing cache.

Winner: TabPFN (max_fit_rows=21119, n_estimators=4, softmax_temperature=0.85)
Policy clip: ±10%

Source: artifacts/features/{snapshot_id}/tabular_features.parquet
        (50k rows — the same table TabPFN was trained on)
Model:  artifacts/models/{snapshot_id}/pricing/tabpfn/model.tabpfn_fit.zip
Output: artifacts/caches/{snapshot_id}/pricing/{pricing_cache.json, .parquet, _manifest.json}

Note: coverage is limited to the ~50k products in tabular_features.parquet.
      See build_pricing_features.py for a full-coverage alternative once the
      original feature engineering pipeline is confirmed by the team.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"
POLICY_BOUND = 10.0

FEATURE_COLS = [
    "price", "avg_rating", "rating_count", "review_count", "review_count_agg",
    "avg_review_rating", "avg_helpful_vote", "positive_ratio", "avg_star_rating",
    "recent_review_ratio_90d", "days_since_last_review", "price_percentile_in_subcategory",
    "rating_vs_subcategory_mean", "subcategory_median_price", "subcategory_mean_price",
    "subcategory_mean_rating", "subcategory_product_count", "inferred_pack_units",
    "price_per_unit", "subcategory_mean_price_per_unit", "on_hand_units",
    "safety_stock_units", "mean_daily_revenue", "total_units_sold", "total_returns",
    "n_sale_days", "log_price", "rating_price_ratio",
]


def run(
    snapshot_id: str = SNAPSHOT_ID,
    artifacts_root: Path | None = None,
) -> Path:
    root = artifacts_root or Path(__file__).resolve().parents[3] / "artifacts"
    src = root / "features" / snapshot_id / "tabular_features.parquet"
    model_path = root / "models" / snapshot_id / "pricing" / "tabpfn" / "model.tabpfn_fit.zip"
    out_dir = root / "caches" / snapshot_id / "pricing"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Tabular features not found at {src}.")
    if not model_path.exists():
        raise FileNotFoundError(f"TabPFN model not found at {model_path}.")

    print(f"Loading {src}")
    df = pd.read_parquet(src)
    product_ids = df["product_id"].astype(str).tolist()

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].fillna(0).astype("float32").values

    print(f"Loading TabPFN model from {model_path}")
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor.load(str(model_path))
    print(f"Predicting on {len(X)} rows...")
    preds = model.predict(X).clip(-POLICY_BOUND, POLICY_BOUND)

    cache_dict = {pid: float(p) for pid, p in zip(product_ids, preds)}
    cache_df = pd.DataFrame(
        {"product_id": product_ids, "predicted_price_change_pct": preds.tolist()}
    )

    (out_dir / "pricing_cache.json").write_text(json.dumps(cache_dict), encoding="utf-8")
    cache_df.to_parquet(out_dir / "pricing_cache.parquet", index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "source_table": str(src.as_posix()).split("cod-multiagent-ecommerce/")[-1],
        "model_state_path": str(model_path.as_posix()).split("cod-multiagent-ecommerce/")[-1],
        "policy_bound": POLICY_BOUND,
        "rows": len(cache_df),
        "columns": list(cache_df.columns),
    }
    (out_dir / "pricing_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(cache_df)} rows → {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default=SNAPSHOT_ID)
    args = parser.parse_args()
    run(snapshot_id=args.snapshot_id)
