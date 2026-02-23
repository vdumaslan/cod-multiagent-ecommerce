from __future__ import annotations

from .contracts import AgentOutput


class PricingAgent:
    def run(self, candidate_ids: list[str]) -> AgentOutput:
        # Placeholder: replace with FT-Transformer inference over tabular pricing features.
        return AgentOutput(
            agent_name="pricing_agent",
            claim="FT-Transformer pricing model favors items with stronger value-for-money.",
            recommended_items=candidate_ids[:3],
            confidence=0.61,
            evidence=["predicted fair-price range", "category price percentile", "rating-to-price ratio"],
            risks_or_limitations=["dynamic market pricing not yet integrated", "model requires periodic retraining"],
            metadata={"model_id": "FT-Transformer"},
        )
