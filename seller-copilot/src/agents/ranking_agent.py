from __future__ import annotations

from .contracts import AgentOutput


class RankingAgent:
    def run(self, candidate_ids: list[str]) -> AgentOutput:
        # Placeholder: replace with reranker inference.
        return AgentOutput(
            agent_name="ranking_agent",
            claim="Top candidates best match query intent and product relevance.",
            recommended_items=candidate_ids[:3],
            confidence=0.67,
            evidence=["semantic similarity", "reranker score"],
            risks_or_limitations=["reranker fine-tuning pending"],
        )

