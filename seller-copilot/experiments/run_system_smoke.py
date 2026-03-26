#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from copilot_runtime.debate import build_ranked_plans
from copilot_runtime.logging_utils import append_decision_log
from copilot_runtime.retrieval import load_retriever, retrieve_evidence


def main() -> None:
    products = pd.read_parquet(ROOT / "data" / "agent_dataset" / "products.parquet")
    signals = pd.read_parquet(ROOT / "data" / "agent_dataset" / "product_signals.parquet")
    inventory = pd.read_parquet(ROOT / "data" / "synthetic" / "inventory_skus.parquet")

    retriever = load_retriever(ROOT / "artifacts" / "faiss")
    query = "Increase revenue while reducing dead inventory"
    evidence = retrieve_evidence(query, retriever, k=8)
    plans, trace, warnings = build_ranked_plans(
        goal=query,
        evidence=evidence,
        products=products,
        signals=signals,
        inventory=inventory,
        min_margin=0.08,
        max_abs_price_change=10.0,
    )

    payload = {
        "query": query,
        "plans": plans,
        "decision": "Approve",
        "reason": "Smoke test auto-approve",
        "latency_ms": 0,
        "models": {"retrieval": retriever["model_name"], "debate": "deterministic_v1", "policy": "rules_v1"},
    }
    append_decision_log(ROOT / "artifacts" / "logs" / "decision_logs.jsonl", payload)

    out = {
        "plans": len(plans),
        "warnings": warnings,
        "trace_rounds": list(trace.keys()),
        "log_path": str(ROOT / "artifacts" / "logs" / "decision_logs.jsonl"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
