from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import AgentOutput
from .llm_runtime import LLMRuntime


class PricingAgent:
    def __init__(
        self,
        pricing_parquet: str = "seller-copilot/artifacts/data/pricing_train.parquet",
        llm_model_id: str | None = None,
        llm_fallback_model_id: str | None = None,
    ) -> None:
        self.stats: dict[str, tuple[float, float]] = {}
        self.llm_model_id = llm_model_id
        self.reasoner = (
            LLMRuntime(model_id=llm_model_id, fallback_model_id=llm_fallback_model_id) if llm_model_id else None
        )
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
        evidence = [f"{pid}: value_score={value:.4f}, avg_price={price:.2f}" for pid, value, price in scored[:3]]
        claim = "Pricing/value signals favor products with higher rating-to-price efficiency."
        llm_used = None
        if self.reasoner is not None and evidence:
            llm_text = self.reasoner.generate(
                system_prompt=(
                    "You are the Pricing agent in a seller decision system. "
                    "Return one concise value/pricing recommendation claim."
                ),
                user_prompt="Pricing evidence:\n" + "\n".join(evidence),
                max_new_tokens=120,
                temperature=0.1,
            )
            if llm_text:
                claim = llm_text.strip()
                llm_used = self.reasoner.last_model_used
        return AgentOutput(
            agent_name="pricing_agent",
            claim=claim,
            recommended_items=top,
            confidence=min(0.95, max(0.5, 0.5 + avg_value)),
            evidence=evidence,
            risks_or_limitations=["Dynamic competitor pricing feed is currently unavailable."],
            metadata={"model_id": "FT-Transformer", "llm_model_id": llm_used},
        )
