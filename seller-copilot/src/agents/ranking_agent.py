from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import AgentOutput


class RankingAgent:
    def __init__(self, ranking_parquet: str = "seller-copilot/artifacts/data/ranking_train.parquet") -> None:
        self.scores: dict[str, float] = {}
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
        return AgentOutput(
            agent_name="ranking_agent",
            claim="Reranking relevance labels prioritize products that align with high-intent queries.",
            recommended_items=top,
            confidence=conf,
            evidence=[f"{pid}: avg_relevance={score:.3f}" for pid, score in ranked[:3]],
            risks_or_limitations=["Current ranking features use coarse query templates."],
        )
