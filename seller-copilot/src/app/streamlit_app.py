from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.discovery_agent import DiscoveryAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.pricing_agent import PricingAgent
from agents.ranking_agent import RankingAgent
from agents.sentiment_agent import SentimentAgent
from agents.contracts import AgentOutput


def _load_models() -> dict:
    path = Path("seller-copilot/config/models.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _model(cfg: dict, section: str, key: str = "primary", default: str | None = None) -> str | None:
    node = cfg.get(section, {})
    if not isinstance(node, dict):
        return default
    value = node.get(key, default)
    return str(value) if value is not None else None


models_cfg = _load_models()

st.set_page_config(page_title="CoD Seller Copilot", layout="wide")
st.title("CoD Multi-Agent E-Commerce (Seller Copilot)")
st.caption("Debate architecture: Discovery + Sentiment + Ranking + Pricing + Orchestrator")

query = st.text_input("What are you looking for?", value="durable sports water bottle")
run = st.button("Run Debate")

if run:
    try:
        discovery = DiscoveryAgent(
            embed_model_id=_model(models_cfg, "embeddings", "primary", "BAAI/bge-large-en-v1.5") or "BAAI/bge-large-en-v1.5",
            index_path="seller-copilot/artifacts/retrieval/retrieval_index.faiss",
            lookup_path="seller-copilot/artifacts/retrieval/retrieval_lookup.parquet",
            analysis_model_id=_model(models_cfg, "discovery_llm", "primary"),
            analysis_fallback_model_id=_model(models_cfg, "discovery_llm", "fallback"),
        )
        candidates_df, discovery_trace = discovery.run(query, top_k=10)
        candidate_ids = candidates_df["product_id"].astype(str).tolist()
    except Exception as exc:
        st.warning(f"Discovery artifacts are unavailable: {exc}")
        candidate_ids = ["candidate_A", "candidate_B", "candidate_C"]
        discovery_trace = AgentOutput(
            agent_name="discovery_agent",
            claim="Discovery artifacts are unavailable; using fallback candidate IDs.",
            recommended_items=candidate_ids[:3],
            confidence=0.0,
            evidence=[],
            risks_or_limitations=["Retrieval index or lookup artifact is missing."],
            metadata={"embedding_model_id": _model(models_cfg, "embeddings", "primary")},
        )

    sentiment = SentimentAgent(
        llm_model_id=_model(models_cfg, "sentiment_llm", "primary"),
        llm_fallback_model_id=_model(models_cfg, "sentiment_llm", "fallback"),
    ).run(candidate_ids)
    ranking = RankingAgent(
        llm_model_id=_model(models_cfg, "ranking_llm", "primary"),
        llm_fallback_model_id=_model(models_cfg, "ranking_llm", "fallback"),
    ).run(candidate_ids)
    pricing = PricingAgent(
        llm_model_id=_model(models_cfg, "pricing_llm", "primary"),
        llm_fallback_model_id=_model(models_cfg, "pricing_llm", "fallback"),
    ).run(candidate_ids)

    traces = [discovery_trace, sentiment, ranking, pricing]
    result = OrchestratorAgent(
        llm_model_id=_model(models_cfg, "orchestrator", "primary"),
        llm_fallback_model_id=_model(models_cfg, "orchestrator", "fallback"),
    ).synthesize(traces)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Decision")
        st.write(f"Winner: `{result.winner}`")
        st.write(f"Runner-up: `{result.runner_up}`")
        st.write(f"Uncertainty: `{result.uncertainty}`")
        st.write(result.rationale)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve Recommendation"):
                st.success("Decision captured: APPROVE (human-in-the-loop).")
        with c2:
            if st.button("Reject Recommendation"):
                st.error("Decision captured: REJECT (human-in-the-loop).")
    with col2:
        st.subheader("Debate Traces")
        for trace in result.traces:
            st.markdown(f"**{trace.agent_name}**")
            st.write(trace.claim)
            st.write(f"Top picks: {trace.recommended_items}")
            st.write(f"Confidence: {trace.confidence}")
            st.write(f"Evidence: {trace.evidence}")
            st.write(f"Risks: {trace.risks_or_limitations}")
            st.write(f"Models: {trace.metadata}")
            st.divider()


