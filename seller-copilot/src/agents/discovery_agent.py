from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .contracts import AgentOutput
from .llm_runtime import LLMRuntime


class DiscoveryAgent:
    def __init__(
        self,
        embed_model_id: str,
        index_path: str,
        lookup_path: str,
        analysis_model_id: str | None = None,
        analysis_fallback_model_id: str | None = None,
    ) -> None:
        self.embed_model_id = embed_model_id
        self.embedder = SentenceTransformer(embed_model_id)
        self.index = faiss.read_index(index_path)
        self.lookup = pd.read_parquet(lookup_path)
        self.reasoner = (
            LLMRuntime(model_id=analysis_model_id, fallback_model_id=analysis_fallback_model_id)
            if analysis_model_id
            else None
        )

    def retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = q.astype("float32")
        scores, idx = self.index.search(q, top_k)
        out = self.lookup.iloc[idx[0]].copy().reset_index(drop=True)
        out["retrieval_score"] = scores[0]
        return out

    def run(self, query: str, top_k: int = 20) -> tuple[pd.DataFrame, AgentOutput]:
        results = self.retrieve(query, top_k=top_k)
        top = results.head(3).copy()
        candidate_ids = top["product_id"].astype(str).tolist() if "product_id" in top.columns else []
        evidence = []
        for row in top.itertuples(index=False):
            pid = getattr(row, "product_id", "")
            score = float(getattr(row, "retrieval_score", 0.0))
            evidence.append(f"{pid}: retrieval_score={score:.4f}")

        claim = "Discovery narrowed to semantically relevant candidates for the seller goal."
        llm_used = None
        if self.reasoner is not None and not top.empty:
            preview = []
            for row in top.head(5).itertuples(index=False):
                pid = getattr(row, "product_id", "")
                doc = str(getattr(row, "product_document", ""))[:260]
                score = float(getattr(row, "retrieval_score", 0.0))
                preview.append(f"- product_id={pid}, score={score:.4f}, text={doc}")
            llm_text = self.reasoner.generate(
                system_prompt=(
                    "You are the Discovery agent in a seller decision system. "
                    "Write one concise claim grounded in retrieval scores."
                ),
                user_prompt=f"Query: {query}\nCandidates:\n" + "\n".join(preview),
                max_new_tokens=120,
                temperature=0.1,
            )
            if llm_text:
                claim = llm_text.strip()
                llm_used = self.reasoner.last_model_used

        output = AgentOutput(
            agent_name="discovery_agent",
            claim=claim,
            recommended_items=candidate_ids,
            confidence=0.75 if candidate_ids else 0.0,
            evidence=evidence,
            risks_or_limitations=["Retrieval quality depends on embedding/index coverage."],
            metadata={
                "embedding_model_id": self.embed_model_id,
                "llm_model_id": llm_used,
            },
        )
        return results, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="seller-copilot/artifacts/retrieval/retrieval_index.faiss")
    parser.add_argument("--lookup", default="seller-copilot/artifacts/retrieval/retrieval_lookup.parquet")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    agent = DiscoveryAgent(args.embed_model, args.index, args.lookup)
    results = agent.retrieve(args.query, top_k=10)
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


