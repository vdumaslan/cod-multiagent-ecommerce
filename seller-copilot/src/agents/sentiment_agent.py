from __future__ import annotations

from .contracts import AgentOutput


class SentimentAgent:
    def run(self, candidate_ids: list[str]) -> AgentOutput:
        # Placeholder: replace with model inference over review evidence.
        return AgentOutput(
            agent_name="sentiment_agent",
            claim="User voice favors products with consistent positive review tone.",
            recommended_items=candidate_ids[:3],
            confidence=0.62,
            evidence=["positive review ratio", "low complaint frequency"],
            risks_or_limitations=["aspect extraction not yet fine-tuned"],
        )

