"""Batch inventory classification over inventory_skus + sales_daily → inventory cache.

Rule-based 4-class classifier: stockout_risk | low_stock | overstocked | healthy

Sources:
  artifacts/synthetic/{snapshot_id}/inventory_skus.parquet  (on_hand_units, safety_stock_units)
  artifacts/synthetic/{snapshot_id}/sales_daily.parquet     (gross_revenue_usd, return_units)

Output:
  artifacts/caches/{snapshot_id}/inventory/inventory_cache.parquet
  artifacts/caches/{snapshot_id}/inventory/inventory_cache.json      (with --write-json)
  artifacts/caches/{snapshot_id}/inventory/inventory_cache_manifest.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"


def _quantile(x: pd.Series, q: float, default: float) -> float:
    try:
        v = float(x.quantile(q))
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _classify_row(row: pd.Series, *, thresholds: dict[str, float]) -> tuple[str, bool, list[str]]:
    """Dataset-aware rule classifier.

    Rules are intentionally simple and percentile-based to avoid degenerate outputs when synthetic
    distributions shift.
    """
    on_hand = float(row.get("on_hand_units", 0) or 0)
    safety = float(row.get("safety_stock_units", 0) or 0)
    available = max(on_hand - safety, 0.0)
    revenue = float(row.get("mean_daily_revenue", 0) or 0)
    returns = float(row.get("total_returns", 0) or 0)

    rules: list[str] = []

    # Stockout risk: fire if there is no sellable stock.
    if available <= 0.0 or on_hand <= safety:
        rules.append("stockout_risk")
        return "stockout_risk", True, rules

    # Low stock: bottom quantile of available_to_sell.
    if available <= thresholds["low_stock_available_to_sell_q10"]:
        rules.append("low_stock_available_to_sell")
        return "low_stock", True, rules

    # Overstock: top quantile of on_hand AND (bottom quantile of revenue OR top quantile of returns).
    if on_hand >= thresholds["overstock_on_hand_q90"] and (
        revenue <= thresholds["overstock_low_revenue_q20"] or returns >= thresholds["overstock_high_returns_q90"]
    ):
        if on_hand >= thresholds["overstock_on_hand_q90"]:
            rules.append("overstock_on_hand_high")
        if revenue <= thresholds["overstock_low_revenue_q20"]:
            rules.append("overstock_low_revenue")
        if returns >= thresholds["overstock_high_returns_q90"]:
            rules.append("overstock_high_returns")
        return "overstocked", True, rules

    return "healthy", False, rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default=SNAPSHOT_ID)
    parser.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    parser.add_argument("--write-json", action="store_true", help="Also write inventory_cache.json mapping product_id->dict.")
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)

    skus_src = root / "synthetic" / snapshot_id / "inventory_skus.parquet"
    sales_src = root / "synthetic" / snapshot_id / "sales_daily.parquet"
    out_dir = root / "caches" / snapshot_id / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {skus_src}")
    skus = pd.read_parquet(skus_src, columns=["product_id", "on_hand_units", "safety_stock_units"])

    print(f"Loading {sales_src}")
    sales = pd.read_parquet(sales_src, columns=["product_id", "gross_revenue_usd", "return_units"])
    sales_agg = sales.groupby("product_id").agg(
        mean_daily_revenue=("gross_revenue_usd", "mean"),
        total_returns=("return_units", "sum"),
    ).reset_index()

    df = skus.merge(sales_agg, on="product_id", how="left").fillna(0)

    # Dataset-aware thresholds to prevent degenerate class distributions.
    available_to_sell = (df["on_hand_units"].astype(float) - df["safety_stock_units"].astype(float)).clip(lower=0.0)
    thresholds = {
        "low_stock_available_to_sell_q10": _quantile(available_to_sell, 0.10, 1.0),
        "overstock_on_hand_q90": _quantile(df["on_hand_units"].astype(float), 0.90, 50.0),
        "overstock_low_revenue_q20": _quantile(df["mean_daily_revenue"].astype(float), 0.20, 1.0),
        "overstock_high_returns_q90": _quantile(df["total_returns"].astype(float), 0.90, 10.0),
    }

    results = []
    for _, row in df.iterrows():
        status, risk, rules = _classify_row(row, thresholds=thresholds)
        on_hand = float(row.get("on_hand_units", 0) or 0)
        safety = float(row.get("safety_stock_units", 0) or 0)
        results.append({
            "product_id": str(row["product_id"]),
            "stock_status": status,
            "risk_flag": risk,
            "on_hand_units": on_hand,
            "safety_stock_units": safety,
            "available_to_sell": max(on_hand - safety, 0.0),
            "mean_daily_revenue": float(row.get("mean_daily_revenue", 0) or 0),
            "total_returns": float(row.get("total_returns", 0) or 0),
            "rules_fired": rules,
        })

    cache_df = pd.DataFrame(results)
    out_path = out_dir / "inventory_cache.parquet"
    cache_df.to_parquet(out_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "source_skus": str(skus_src),
        "source_sales": str(sales_src),
        "approach": "rule_based",
        "thresholds": {
            **thresholds,
        },
        "classes": ["stockout_risk", "low_stock", "overstocked", "healthy"],
        "rows": len(cache_df),
        "columns": list(cache_df.columns),
    }
    (out_dir / "inventory_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if bool(args.write_json):
        cache_dict = {
            r["product_id"]: {k: v for k, v in r.items() if k != "product_id"}
            for r in results
        }
        (out_dir / "inventory_cache.json").write_text(json.dumps(cache_dict), encoding="utf-8")

    dist = cache_df["stock_status"].value_counts().to_dict()
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "parquet": str(out_path), "rows": len(cache_df), "distribution": dist}))


if __name__ == "__main__":
    main()
