#!/usr/bin/env python3
"""
Lightweight policy backtest on synthetic sales history.

Goal: estimate whether recommended SKUs have better next-14-day revenue trend
than a random SKU basket baseline. This is a quick accuracy sanity check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copilot_runtime.debate import build_ranked_plans
from copilot_runtime.retrieval import load_retriever, retrieve_evidence


def _basket_future_lift(sales: pd.DataFrame, skus: list[str], split_date: pd.Timestamp) -> float:
    pre0 = split_date - pd.Timedelta(days=14)
    post1 = split_date + pd.Timedelta(days=14)
    pre = sales[(sales["sale_date"] >= pre0) & (sales["sale_date"] < split_date) & (sales["product_id"].isin(skus))]
    post = sales[(sales["sale_date"] >= split_date) & (sales["sale_date"] < post1) & (sales["product_id"].isin(skus))]
    pre_rev = float(pre["gross_revenue_usd"].sum())
    post_rev = float(post["gross_revenue_usd"].sum())
    if pre_rev <= 1e-9:
        return 0.0
    return (post_rev - pre_rev) / pre_rev


def main() -> None:
    products = pd.read_parquet(ROOT / "data" / "agent_dataset" / "products.parquet")
    signals = pd.read_parquet(ROOT / "data" / "agent_dataset" / "product_signals.parquet")
    inventory = pd.read_parquet(ROOT / "data" / "synthetic" / "inventory_skus.parquet")
    sales = pd.read_parquet(ROOT / "data" / "synthetic" / "sales_daily.parquet")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])

    retriever = load_retriever(ROOT / "artifacts" / "faiss")

    # Representative queries for business goals.
    queries = [
        "increase revenue while protecting margin",
        "reduce dead inventory with minimal margin loss",
        "improve customer satisfaction and repeat purchases",
        "cut costs in low-performing SKUs",
        "grow sales in fast-moving products",
    ]

    # Split date in synthetic period midpoint.
    split_date = sales["sale_date"].min() + (sales["sale_date"].max() - sales["sale_date"].min()) / 2

    rng = np.random.default_rng(42)
    universe = products["product_id"].tolist()

    lifts_model: list[float] = []
    lifts_random: list[float] = []

    for q in queries:
        evidence = retrieve_evidence(q, retriever, k=10, candidate_k=80, dense_weight=0.7, lexical_weight=0.3)
        plans, _, _ = build_ranked_plans(
            goal=q,
            evidence=evidence,
            products=products,
            signals=signals,
            inventory=inventory,
            min_margin=0.08,
            max_abs_price_change=10.0,
        )
        sku_set: list[str] = []
        for p in plans:
            sku_set.extend(p.get("impacted_skus", []))
        sku_set = list(dict.fromkeys(sku_set))[:9]
        if not sku_set:
            continue

        lift_model = _basket_future_lift(sales, sku_set, split_date)
        random_skus = rng.choice(universe, size=min(len(sku_set), len(universe)), replace=False).tolist()
        lift_rand = _basket_future_lift(sales, random_skus, split_date)

        lifts_model.append(lift_model)
        lifts_random.append(lift_rand)

    payload = {
        "n_queries": len(lifts_model),
        "mean_lift_model": float(np.mean(lifts_model)) if lifts_model else 0.0,
        "mean_lift_random": float(np.mean(lifts_random)) if lifts_random else 0.0,
        "delta_vs_random": float(np.mean(lifts_model) - np.mean(lifts_random)) if lifts_model else 0.0,
        "note": "Synthetic backtest proxy. Positive delta suggests better SKU targeting than random baseline.",
    }

    out_dir = ROOT / "artifacts" / "evals" / "system"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_backtest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
