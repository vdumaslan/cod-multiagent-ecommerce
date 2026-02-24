from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import AgentOutput


class PricingAgent:
    def __init__(self, pricing_parquet: str = "seller-copilot/artifacts/data/pricing_train.parquet") -> None:
        self.stats: dict[str, tuple[float, float]] = {}
        p = Path(pricing_parquet)
        if p.exists():
            df = pd.read_parquet(p)
            needed = {"product_id", "price", "rating_price_ratio"}
            if needed.issubset(df.columns):
                agg = df.groupby("product_id").agg(
                    avg_price=("price", "mean"),
                    value_score=("rating_price_ratio", "mean"),
                )
                self.stats = {
                    str(idx): (float(row["avg_price"]), float(row["value_score"])) for idx, row in agg.iterrows()
                }

    def run(self, candidate_ids: list[str]) -> AgentOutput:
        if not candidate_ids:
            return AgentOutput(
                agent_name="pricing_agent",
                claim="No candidate products provided.",
                confidence=0.0,
                evidence=[],
                risks_or_limitations=["No retrieval candidates"],
                metadata={"model_id": "FT-Transformer"},
            )

        scored = []
        for cid in candidate_ids:
            price, value = self.stats.get(str(cid), (0.0, 0.0))
            scored.append((cid, value, price))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [cid for cid, _, _ in scored[:3]]
        avg_value = float(sum(v for _, v, _ in scored[:3]) / max(1, len(scored[:3])))
        return AgentOutput(
            agent_name="pricing_agent",
            claim="Pricing/value signals favor products with higher rating-to-price efficiency.",
            recommended_items=top,
            confidence=min(0.95, max(0.5, 0.5 + avg_value)),
            evidence=[f"{pid}: value_score={value:.4f}, avg_price={price:.2f}" for pid, value, price in scored[:3]],
            risks_or_limitations=["Dynamic competitor pricing feed is currently unavailable."],
            metadata={"model_id": "FT-Transformer"},
        )
