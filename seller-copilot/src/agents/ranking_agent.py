from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import AgentOutput
from .llm_runtime import LLMRuntime


class RankingAgent:
    def __init__(
        self,
        ranking_parquet: str = "seller-copilot/artifacts/data/ranking_train.parquet",
        llm_model_id: str | None = None,
        llm_fallback_model_id: str | None = None,
    ) -> None:
        self.scores: dict[str, float] = {}
        self.llm_model_id = llm_model_id
        self.reasoner = (
            LLMRuntime(model_id=llm_model_id, fallback_model_id=llm_fallback_model_id) if llm_model_id else None
        )
        p = Path(ranking_parquet)
        if p.exists():
            df = pd.read_parquet(p)
            if {"product_id", "relevance_label"}.issubset(df.columns):
                agg = df.groupby("product_id")["relevance_label"].mean()
                self.scores = {str(k): float(v) for k, v in agg.items()}

    def run(self, candidate_ids: list[str]) -> AgentOutput:
        if not candidate_ids:
            return AgentOutput(
                agent_name="ranking_agent",
                claim="No candidate products provided.",
                confidence=0.0,
                evidence=[],
                risks_or_limitations=["No retrieval candidates"],
            )

        ranked = sorted(
            [(cid, self.scores.get(str(cid), 0.0)) for cid in candidate_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        top = [cid for cid, _ in ranked[:3]]
        conf = max(0.5, min(0.95, float(sum(s for _, s in ranked[:3]) / max(1, len(ranked[:3])))))
        evidence = [f"{pid}: avg_relevance={score:.3f}" for pid, score in ranked[:3]]
        claim = "Reranking relevance labels prioritize products that align with high-intent queries."
        llm_used = None
        if self.reasoner is not None and evidence:
            llm_text = self.reasoner.generate(
                system_prompt=(
                    "You are the Ranking agent in a seller decision system. "
                    "Return one concise ranking claim grounded in the evidence."
                ),
                user_prompt="Top ranking stats:\n" + "\n".join(evidence),
                max_new_tokens=120,
                temperature=0.1,
            )
            if llm_text:
                claim = llm_text.strip()
                llm_used = self.reasoner.last_model_used
        return AgentOutput(
            agent_name="ranking_agent",
            claim=claim,
            recommended_items=top,
            confidence=conf,
            evidence=evidence,
            risks_or_limitations=["Current ranking features use coarse query templates."],
            metadata={"reranker_model_id": "BAAI/bge-reranker-v2-m3", "llm_model_id": llm_used},
        )
