from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from copilot_runtime.logging_utils import append_decision_log
from copilot_runtime.retrieval import load_retriever, retrieve_evidence

DATA_AGENT = ROOT / "data" / "agent_dataset"
DATA_SYN = ROOT / "data" / "synthetic"
ART = ROOT / "artifacts"
LOG_PATH = ART / "logs" / "decision_logs.jsonl"


def _plain_english_action(action: str) -> str:
    txt = " ".join(str(action).split())
    low = txt.lower()
    if low.startswith("focus on your stated goal:"):
        rest = txt.split(":", 1)[1].strip()
        return f"Keep this goal in mind: {rest}"
    if low.startswith("secondary push aligned to:"):
        rest = txt.split(":", 1)[1].strip()
        return f"Supporting plan for: {rest}"
    if low.startswith("if inventory pressure conflicts with:"):
        rest = txt.split(":", 1)[1].strip()
        return f"If stock issues clash with your goal ({rest}), prioritize clearance carefully."
    if "raise price 3-6%" in low and "hero skus" in low:
        return (
            "Raise the price by 3-6% on your top-performing products with strong reviews, "
            "and move ad spend to products that already convert well."
        )
    if "increase visibility for top sentiment+margin skus" in low:
        return "Promote products that customers like and that give you healthier profit margins."
    if "capped price uplift for top 3 skus" in low:
        return "Apply a small, controlled price increase on your top 3 products."
    if "bundle medium-margin skus" in low:
        return "Bundle mid-margin products together to increase average order value."
    if "shift spend from low-conversion skus" in low:
        return "Reduce ad spend on products that do not convert and reallocate to winners."
    if "clear low-velocity skus with conservative markdown" in low:
        return "Offer small discounts on slow-moving products to clear inventory without heavy margin loss."
    return txt[0].upper() + txt[1:] if txt else "Take action based on plan details."


def format_quick_answer(goal: str, plans: list[dict], products: pd.DataFrame) -> str:
    """One-screen prose summary; structured JSON plans stay below for audit."""
    if not plans:
        return "_No plans produced._"
    p0 = plans[0]
    pct = float(p0.get("price_change_pct", 0.0))
    actions = p0.get("actions") or []
    skus = p0.get("impacted_skus") or []
    impact = (p0.get("expected_impact") or "").strip()
    risk = (p0.get("risk") or "").strip()
    conf = float(p0.get("confidence", 0.0))
    titles: list[str] = []
    for pid in skus[:3]:
        m = products.loc[products["product_id"] == pid]
        if not m.empty:
            t = str(m.iloc[0].get("title", pid))
            titles.append(t[:90] + ("..." if len(t) > 90 else ""))
    action_lines = [_plain_english_action(str(a)) for a in actions] if actions else ["See ranked plans."]
    action_block = "\n".join(f"- {a}" for a in action_lines)
    parts = [
        "**Suggested move (Plan A):**",
        action_block,
        f"**Price lever:** **{pct:+.1f}%** (after policy rules vs your max change slider).",
    ]
    if titles:
        parts.append("**Examples:**\n" + "\n".join(f"- {t}" for t in titles))
    if impact:
        parts.append(f"**Expected impact:** {impact}")
    if risk:
        parts.append(f"**Main risk:** {risk}")
    parts.append(f"**Confidence:** {conf:.0%}")
    parts.append(
        "_Below: matched products, optional technical detail, then ranked plans and approve/reject._"
    )
    return "\n\n".join(parts)


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "products": pd.read_parquet(DATA_AGENT / "products.parquet"),
        "signals": pd.read_parquet(DATA_AGENT / "product_signals.parquet"),
        "inventory": pd.read_parquet(DATA_SYN / "inventory_skus.parquet"),
    }


st.set_page_config(page_title="Business Co-Pilot", layout="wide")
st.title("Business Co-Pilot")
st.caption("Business decision support for e-commerce: goals, evidence-backed plans, approve/reject.")
st.info("The first run can take a minute while models load. Turn off **Use AI to phrase plans** below for faster results.")

with st.sidebar:
    st.markdown("### Policy constraints")
    min_margin = st.slider("Min margin floor", 0.0, 0.5, 0.08, 0.01)
    max_change = st.slider("Max |price change| %", 1.0, 25.0, 10.0, 1.0)
    st.markdown("### Product matching")
    dense_w = st.slider(
        "Match style (keywords ↔ meaning)",
        0.1,
        0.9,
        0.7,
        0.05,
        help="Higher = match by meaning. Lower = match exact words in titles and descriptions.",
    )
    lexical_w = round(1.0 - dense_w, 2)
    st.caption("Balances keyword vs meaning-based matching.")
    use_market_reranker = st.toggle("Refine news headlines", value=True)
    use_llm_orch = st.toggle("Use AI to phrase plans", value=False)

query = st.text_input("Business goal", value="Increase revenue while protecting margin")
run = st.button("Generate plans")

if run:
    from copilot_runtime.debate import build_ranked_plans

    t0 = time.time()
    with st.spinner("Preparing your recommendations…"):
        tables = load_tables()
        retriever = load_retriever(ART / "faiss")
        evidence = retrieve_evidence(
            query,
            retriever,
            k=10,
            candidate_k=80,
            dense_weight=dense_w,
            lexical_weight=lexical_w,
        )
        plans, trace, warnings = build_ranked_plans(
            goal=query,
            evidence=evidence,
            products=tables["products"],
            signals=tables["signals"],
            inventory=tables["inventory"],
            min_margin=min_margin,
            max_abs_price_change=max_change,
            use_market_reranker=use_market_reranker,
            use_llm_orchestrator=use_llm_orch,
        )
    latency_ms = int((time.time() - t0) * 1000)

    st.markdown(format_quick_answer(query, plans, tables["products"]))

    st.subheader("Matched products")
    ev_show = evidence.head(10)[["product_id", "score"]].copy()
    ev_show = ev_show.rename(columns={"score": "match_score"})
    st.dataframe(ev_show, width="stretch", hide_index=True)
    with st.expander("Technical detail (scores)", expanded=False):
        st.dataframe(
            evidence[["product_id", "score", "score_dense", "score_lexical"]].head(10),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Technical detail (agent trace)", expanded=False):
        st.json(trace)

    for w in warnings:
        st.warning(w)

    st.subheader("Recommended plans")
    for p in plans:
        with st.expander(f"Rank {p['rank']} - {p['plan_id']}", expanded=(p["rank"] == 1)):
            st.write("Actions:", p["actions"])
            st.write("Impacted SKUs:", p["impacted_skus"])
            st.write("Expected impact:", p["expected_impact"])
            st.write("Risk:", p["risk"])
            st.write("Confidence:", p["confidence"])
            st.write("Evidence refs:", p["evidence_refs"])

    st.subheader("Decision")
    choice = st.radio("Approve or reject", ["Approve", "Reject"], horizontal=True)
    reason = st.text_input("Reason (optional)")
    if st.button("Save decision"):
        payload = {
            "query": query,
            "plans": plans,
            "decision": choice,
            "reason": reason,
            "latency_ms": latency_ms,
            "models": {
                "retrieval": retriever["model_name"],
                "debate": "deterministic_v1",
                "policy": "rules_v1",
                "market_reranker": (
                    trace.get("market_model", {}).get("model_name")
                    if trace.get("market_model", {}).get("enabled")
                    else "disabled"
                ),
                "orchestrator": (
                    trace.get("orchestrator_model", {}).get("model_name")
                    if trace.get("orchestrator_model", {}).get("enabled")
                    else "deterministic_v1"
                ),
            },
        }
        append_decision_log(LOG_PATH, payload)
        st.success(f"Decision logged to {LOG_PATH}")
