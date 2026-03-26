from __future__ import annotations

from typing import Any

import pandas as pd

from .market_intel import (
    fetch_market_news,
    market_inventory_suggestions,
    rerank_market_news,
    summarize_market_signal,
)
from .policy import apply_policy


def _goal_preview(goal: str, max_len: int = 120) -> str:
    g = " ".join(goal.split())
    if len(g) <= max_len:
        return g
    return g[: max_len - 1] + "…"


def _llm_action_is_placeholder(a: object) -> bool:
    s = str(a).strip().lower()
    return s in ("", "string", "...", "n/a", "none", "tbd")


def _merge_llm_with_deterministic(
    norm: list[dict[str, Any]], deterministic_plans: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep LLM SKU picks when sane; always repair empty/placeholder text from deterministic templates."""
    out: list[dict[str, Any]] = []
    for i in range(3):
        d = deterministic_plans[i]
        n = dict(norm[i]) if i < len(norm) else dict(d)
        acts = n.get("actions") or []
        if not acts or all(_llm_action_is_placeholder(x) for x in acts):
            n["actions"] = list(d["actions"])
        else:
            cleaned = [str(x) for x in acts if not _llm_action_is_placeholder(x)]
            n["actions"] = cleaned if cleaned else list(d["actions"])
        if not str(n.get("expected_impact", "")).strip():
            n["expected_impact"] = d["expected_impact"]
        if not str(n.get("risk", "")).strip():
            n["risk"] = d["risk"]
        conf = float(n.get("confidence", 0.0))
        if conf < 0.05 or conf > 1.0:
            n["confidence"] = float(d["confidence"])
        else:
            n["confidence"] = conf
        try:
            pcp = float(n.get("price_change_pct", d["price_change_pct"]))
        except (TypeError, ValueError):
            pcp = float(d["price_change_pct"])
        n["price_change_pct"] = pcp
        n["plan_id"] = n.get("plan_id") or d["plan_id"]
        n["rank"] = i + 1
        if not n.get("impacted_skus"):
            n["impacted_skus"] = list(d["impacted_skus"])
        if not n.get("evidence_refs"):
            n["evidence_refs"] = list(d["evidence_refs"])
        out.append(n)
    return out


def build_ranked_plans(
    *,
    goal: str,
    evidence: pd.DataFrame,
    products: pd.DataFrame,
    signals: pd.DataFrame,
    inventory: pd.DataFrame,
    min_margin: float,
    max_abs_price_change: float,
    use_market_reranker: bool = True,
    use_llm_orchestrator: bool = False,
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    top = evidence.merge(products[["product_id", "title", "price", "avg_rating"]], on="product_id", how="left")
    top = top.merge(signals[["product_id", "positive_ratio", "recency_score"]], on="product_id", how="left")
    top = top.merge(inventory[["product_id", "margin_pct", "available_to_sell"]], on="product_id", how="left")
    market_news_raw = fetch_market_news(goal, limit=20)
    market_news, market_model = rerank_market_news(
        goal,
        market_news_raw,
        top_n=10,
        use_model=use_market_reranker,
    )
    market_signal = summarize_market_signal(goal, market_news)
    market_skus = market_inventory_suggestions(top, market_signal, max_skus=6)

    # Query-relevant ordering: hybrid retrieval score first, then recency (deterministic mode).
    by_score = top.sort_values(["score", "recency_score"], ascending=[False, False])
    by_margin = top.sort_values(["margin_pct", "avg_rating"], ascending=[False, False])
    by_stock_risk = top.sort_values(["available_to_sell", "margin_pct"], ascending=[True, False])

    round1 = [
        {
            "agent": "demand_signals",
            "argument": f"Prioritize retrieval-matched SKUs (hybrid score) for goal: {goal}",
            "impacted_skus": by_score["product_id"].head(6).tolist(),
            "confidence": 0.72,
            "evidence_refs": by_score["product_id"].head(6).tolist(),
        },
        {
            "agent": "pricing_margin",
            "argument": "Prefer healthy margin SKUs with stable ratings among retrieved evidence.",
            "impacted_skus": by_margin["product_id"].head(6).tolist(),
            "confidence": 0.70,
            "evidence_refs": by_score["product_id"].head(6).tolist(),
        },
        {
            "agent": "inventory_ops",
            "argument": "Avoid stockout risk by filtering low available inventory among retrieved evidence.",
            "impacted_skus": by_stock_risk["product_id"].head(6).tolist(),
            "confidence": 0.68,
            "evidence_refs": by_score["product_id"].head(6).tolist(),
        },
        {
            "agent": "market_intelligence",
            "argument": (
                "Incorporate live market/news trend with inventory context; "
                f"trend_score={market_signal.get('trend_score', 0.0):.2f}."
            ),
            "impacted_skus": market_skus,
            "confidence": 0.66 + 0.20 * float(market_signal.get("trend_score", 0.0)),
            "evidence_refs": by_score["product_id"].head(6).tolist(),
            "market_signal": market_signal,
        },
    ]

    round2 = [{**a, "rebuttal": "Adjusted for inventory and margin trade-offs.", "confidence": min(0.9, a["confidence"] + 0.04)} for a in round1]

    all_skus: list[str] = []
    all_evidence: list[str] = []
    for r in round2:
        all_skus.extend(r["impacted_skus"][:3])
        all_evidence.extend(r["evidence_refs"][:3])
    unique_skus = list(dict.fromkeys(all_skus))[:9]
    unique_ev = list(dict.fromkeys(all_evidence))[:9]

    preview = _goal_preview(goal)
    deterministic_plans = [
        {
            "plan_id": "plan_a",
            "rank": 1,
            "actions": [
                f"Focus on your stated goal: {preview}",
                "Increase visibility for top sentiment+margin SKUs among retrieved matches",
                "Capped price uplift for top 3 SKUs",
            ],
            "impacted_skus": unique_skus[:3],
            "expected_impact": "Revenue + margin uplift",
            "risk": "Price elasticity risk",
            "confidence": 0.78,
            "evidence_refs": unique_ev[:4],
            "price_change_pct": 6.0,
        },
        {
            "plan_id": "plan_b",
            "rank": 2,
            "actions": [
                f"Secondary push aligned to: {preview}",
                "Bundle medium-margin SKUs",
                "Shift spend from low-conversion SKUs",
            ],
            "impacted_skus": unique_skus[3:6],
            "expected_impact": "AOV growth with stable sell-through",
            "risk": "Execution complexity",
            "confidence": 0.73,
            "evidence_refs": unique_ev[2:6],
            "price_change_pct": 4.0,
        },
        {
            "plan_id": "plan_c",
            "rank": 3,
            "actions": [
                f"If inventory pressure conflicts with: {preview}",
                "Clear low-velocity SKUs with conservative markdown",
            ],
            "impacted_skus": unique_skus[6:9],
            "expected_impact": "Inventory relief",
            "risk": "Lower short-term margin",
            "confidence": 0.69,
            "evidence_refs": (unique_ev[4:9] if unique_ev[4:9] else unique_ev[:2]),
            "price_change_pct": -7.0,
        },
    ]
    plans = deterministic_plans
    orchestrator_model = {"enabled": False, "model_name": None, "status": "deterministic_only"}
    if use_llm_orchestrator:
        from .llm_orchestrator import generate_llm_plans

        ev_snips: list[dict[str, str]] = []
        for _, r in by_score.head(10).iterrows():
            ev_snips.append(
                {
                    "product_id": str(r["product_id"]),
                    "title": str(r.get("title", ""))[:180],
                    "snippet": str(r.get("product_document", ""))[:300],
                }
            )

        llm_plans, orchestrator_model = generate_llm_plans(
            model_name=llm_model_name,
            goal=goal,
            round2=round2,
            evidence_ids=unique_ev,
            evidence_snippets=ev_snips,
        )
        if llm_plans:
            # Sanitize/normalize LLM output to required schema and safe ids.
            allowed_ids = set(unique_skus)
            allowed_ev = set(unique_ev)
            norm: list[dict[str, Any]] = []
            for i, p in enumerate(llm_plans[:3], start=1):
                try:
                    confidence = float(p.get("confidence", 0.7))
                except (TypeError, ValueError):
                    confidence = 0.7
                try:
                    price_change_pct = float(p.get("price_change_pct", 0.0))
                except (TypeError, ValueError):
                    price_change_pct = 0.0
                raw_skus = p.get("impacted_skus", [])
                raw_ev = p.get("evidence_refs", [])
                skus = [s for s in raw_skus if s in allowed_ids][:3]
                evs = [e for e in raw_ev if e in allowed_ev][:4]
                if not skus:
                    skus = unique_skus[(i - 1) * 3 : i * 3] if unique_skus else []
                if not evs:
                    evs = unique_ev[(i - 1) * 2 : i * 2] if unique_ev else []
                norm.append(
                    {
                        "plan_id": p.get("plan_id", f"plan_{chr(96 + i)}"),
                        "rank": i,
                        "actions": p.get("actions", ["No-op"]),
                        "impacted_skus": skus,
                        "expected_impact": p.get("expected_impact", "Business impact TBD"),
                        "risk": p.get("risk", "Execution risk"),
                        "confidence": confidence,
                        "evidence_refs": evs,
                        "price_change_pct": price_change_pct,
                    }
                )
            if norm:
                plans = _merge_llm_with_deterministic(norm, deterministic_plans)

    final_plans, warnings = apply_policy(
        plans,
        inventory,
        min_margin=min_margin,
        max_abs_price_change=max_abs_price_change,
    )
    qx = ""
    if len(evidence) and "query_expanded" in evidence.columns:
        qx = str(evidence["query_expanded"].iloc[0])
    trace = {
        "goal": goal,
        "query_expanded": qx or goal,
        "round1": round1,
        "round2": round2,
        "market_signal": market_signal,
        "market_news": market_news[:5],
        "market_model": market_model,
        "orchestrator_model": orchestrator_model,
    }
    return final_plans, trace, warnings
