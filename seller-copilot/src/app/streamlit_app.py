from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.discovery_agent import DiscoveryAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.pricing_agent import PricingAgent
from agents.ranking_agent import RankingAgent
from agents.sentiment_agent import SentimentAgent


st.set_page_config(page_title="CoD Seller Copilot", layout="wide")
st.title("CoD Multi-Agent E-Commerce (Seller Copilot)")
st.caption("Debate architecture: Discovery + Sentiment + Ranking + Pricing + Orchestrator")

query = st.text_input("What are you looking for?", value="durable sports water bottle")
run = st.button("Run Debate")

if run:
    try:
        discovery = DiscoveryAgent(
            embed_model_id="BAAI/bge-large-en-v1.5",
            index_path="seller-copilot/artifacts/retrieval/retrieval_index.faiss",
            lookup_path="seller-copilot/artifacts/retrieval/retrieval_lookup.parquet",
        )
        candidates_df = discovery.retrieve(query, top_k=10)
        candidate_ids = candidates_df["product_id"].astype(str).tolist()
    except Exception as exc:
        st.warning(f"Discovery artifacts not ready yet: {exc}")
        candidate_ids = ["candidate_A", "candidate_B", "candidate_C"]

    sentiment = SentimentAgent().run(candidate_ids)
    ranking = RankingAgent().run(candidate_ids)
    pricing = PricingAgent().run(candidate_ids)

    result = OrchestratorAgent().synthesize([sentiment, ranking, pricing])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Decision")
        st.write(f"Winner: `{result.winner}`")
        st.write(f"Runner-up: `{result.runner_up}`")
        st.write(f"Uncertainty: `{result.uncertainty}`")
        st.write(result.rationale)
    with col2:
        st.subheader("Debate Traces")
        for trace in result.traces:
            st.markdown(f"**{trace.agent_name}**")
            st.write(trace.claim)
            st.write(f"Top picks: {trace.recommended_items}")
            st.write(f"Confidence: {trace.confidence}")
            st.write(f"Evidence: {trace.evidence}")
            st.write(f"Risks: {trace.risks_or_limitations}")
            st.divider()


