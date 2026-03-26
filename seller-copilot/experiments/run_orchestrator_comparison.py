#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
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


def _run_mode(name: str, use_llm_orchestrator: bool, model_name: str) -> dict:
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

    lifts: list[float] = []
    latencies: list[float] = []
    valid_plans = 0
    total_plans = 0
    top1_conf: list[float] = []
    model_statuses: list[str] = []

    for q in queries:
        evidence = retrieve_evidence(q, retriever, k=10, candidate_k=80, dense_weight=0.7, lexical_weight=0.3)
        t0 = time.perf_counter()
        plans, trace, _ = build_ranked_plans(
            goal=q,
            evidence=evidence,
            products=products,
            signals=signals,
            inventory=inventory,
            min_margin=0.08,
            max_abs_price_change=10.0,
            use_market_reranker=True,
            use_llm_orchestrator=use_llm_orchestrator,
            llm_model_name=model_name,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        model_statuses.append(str(trace.get("orchestrator_model", {}).get("status", "na")))

        for p in plans:
            total_plans += 1
            if p.get("actions") and p.get("impacted_skus") and p.get("evidence_refs"):
                valid_plans += 1
        if plans:
            top1_conf.append(float(plans[0].get("confidence", 0.0)))

        skus = list(dict.fromkeys([s for p in plans for s in p.get("impacted_skus", [])]))[:9]
        lifts.append(_basket_future_lift(sales, skus, split_date))

    return {
        "mode": name,
        "n_queries": len(queries),
        "mean_lift": float(np.mean(lifts)) if lifts else 0.0,
        "valid_plan_rate": float(valid_plans / total_plans) if total_plans else 0.0,
        "avg_top1_confidence": float(np.mean(top1_conf)) if top1_conf else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "orchestrator_statuses": model_statuses,
    }


def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    before = _run_mode("deterministic_before", use_llm_orchestrator=False, model_name=model_name)
    after = _run_mode("llm_after", use_llm_orchestrator=True, model_name=model_name)

    comparison = {
        "model_name": model_name,
        "before": before,
        "after": after,
        "delta": {
            "mean_lift": after["mean_lift"] - before["mean_lift"],
            "valid_plan_rate": after["valid_plan_rate"] - before["valid_plan_rate"],
            "avg_top1_confidence": after["avg_top1_confidence"] - before["avg_top1_confidence"],
            "p50_latency_ms": after["p50_latency_ms"] - before["p50_latency_ms"],
            "p95_latency_ms": after["p95_latency_ms"] - before["p95_latency_ms"],
        },
    }

    out_dir = ROOT / "artifacts" / "evals" / "orchestrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
