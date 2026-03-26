#!/usr/bin/env python3
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
from copilot_runtime.market_intel import (
    fetch_market_news,
    rerank_market_news,
    summarize_market_signal,
)
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


def _keyword_precision(goal: str, items: list[dict], top_k: int = 5) -> float:
    tokens = [t.strip(" ,.;:!?()[]{}\"'").lower() for t in goal.split()]
    kws = [t for t in tokens if len(t) >= 4][:6]
    top = items[:top_k]
    if not top:
        return 0.0
    hits = 0
    for it in top:
        title = str(it.get("title", "")).lower()
        if any(k in title for k in kws):
            hits += 1
    return hits / len(top)


def main() -> None:
    products = pd.read_parquet(ROOT / "data" / "agent_dataset" / "products.parquet")
    signals = pd.read_parquet(ROOT / "data" / "agent_dataset" / "product_signals.parquet")
    inventory = pd.read_parquet(ROOT / "data" / "synthetic" / "inventory_skus.parquet")
    sales = pd.read_parquet(ROOT / "data" / "synthetic" / "sales_daily.parquet")
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    split_date = sales["sale_date"].min() + (sales["sale_date"].max() - sales["sale_date"].min()) / 2

    retriever = load_retriever(ROOT / "artifacts" / "faiss")

    queries = [
        "increase revenue while protecting margin",
        "reduce dead inventory with minimal margin loss",
        "improve customer satisfaction and repeat purchases",
        "cut costs in low-performing SKUs",
        "grow sales in fast-moving products",
    ]

    out_rows = []
    for q in queries:
        evidence = retrieve_evidence(q, retriever, k=10, candidate_k=80, dense_weight=0.7, lexical_weight=0.3)

        plans_base, trace_base, _ = build_ranked_plans(
            goal=q,
            evidence=evidence,
            products=products,
            signals=signals,
            inventory=inventory,
            min_margin=0.08,
            max_abs_price_change=10.0,
            use_market_reranker=False,
        )
        plans_model, trace_model, _ = build_ranked_plans(
            goal=q,
            evidence=evidence,
            products=products,
            signals=signals,
            inventory=inventory,
            min_margin=0.08,
            max_abs_price_change=10.0,
            use_market_reranker=True,
        )

        skus_base = list(dict.fromkeys([s for p in plans_base for s in p.get("impacted_skus", [])]))[:9]
        skus_model = list(dict.fromkeys([s for p in plans_model for s in p.get("impacted_skus", [])]))[:9]

        lift_base = _basket_future_lift(sales, skus_base, split_date)
        lift_model = _basket_future_lift(sales, skus_model, split_date)

        raw_news = fetch_market_news(q, limit=20)
        top_base, _ = rerank_market_news(q, raw_news, top_n=10, use_model=False)
        top_model, _ = rerank_market_news(q, raw_news, top_n=10, use_model=True)
        signal_base = summarize_market_signal(q, top_base)
        signal_model = summarize_market_signal(q, top_model)

        out_rows.append(
            {
                "query": q,
                "lift_base": lift_base,
                "lift_model": lift_model,
                "keyword_precision_base": _keyword_precision(q, top_base, top_k=5),
                "keyword_precision_model": _keyword_precision(q, top_model, top_k=5),
                "trend_score_base": float(signal_base.get("trend_score", 0.0)),
                "trend_score_model": float(signal_model.get("trend_score", 0.0)),
            }
        )

    df = pd.DataFrame(out_rows)
    summary = {
        "n_queries": int(len(df)),
        "mean_lift_base": float(df["lift_base"].mean()),
        "mean_lift_model": float(df["lift_model"].mean()),
        "lift_improvement": float(df["lift_model"].mean() - df["lift_base"].mean()),
        "mean_keyword_precision_base": float(df["keyword_precision_base"].mean()),
        "mean_keyword_precision_model": float(df["keyword_precision_model"].mean()),
        "keyword_precision_improvement": float(
            df["keyword_precision_model"].mean() - df["keyword_precision_base"].mean()
        ),
        "mean_trend_score_base": float(df["trend_score_base"].mean()),
        "mean_trend_score_model": float(df["trend_score_model"].mean()),
    }

    out_dir = ROOT / "artifacts" / "evals" / "market_intel"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_rows.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
