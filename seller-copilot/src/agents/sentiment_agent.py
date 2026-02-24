from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import AgentOutput


class SentimentAgent:
    def __init__(self, sentiment_parquet: str = "seller-copilot/artifacts/data/sentiment_train.parquet") -> None:
        self.stats: dict[str, tuple[float, int]] = {}
        p = Path(sentiment_parquet)
        if p.exists():
            df = pd.read_parquet(p)
            if {"product_id", "label"}.issubset(df.columns):
                agg = (
                    df.assign(is_positive=(df["label"].astype(int) == 2).astype(int))
                    .groupby("product_id")
                    .agg(positive_ratio=("is_positive", "mean"), n=("is_positive", "size"))
                )
                self.stats = {
                    str(idx): (float(row["positive_ratio"]), int(row["n"]))
                    for idx, row in agg.iterrows()
                }

    def run(self, candidate_ids: list[str]) -> AgentOutput:
        if not candidate_ids:
            return AgentOutput(
                agent_name="sentiment_agent",
                claim="No candidate products provided.",
                confidence=0.0,
                evidence=[],
                risks_or_limitations=["No retrieval candidates"],
            )

        scored = []
        for cid in candidate_ids:
            ratio, n = self.stats.get(str(cid), (0.5, 0))
            scored.append((cid, ratio, n))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top = [s[0] for s in scored[:3]]
        avg_ratio = float(sum(s[1] for s in scored[:3]) / max(1, len(scored[:3])))

        return AgentOutput(
            agent_name="sentiment_agent",
            claim="User feedback favors candidates with stronger positive sentiment share.",
            recommended_items=top,
            confidence=min(0.95, max(0.5, avg_ratio)),
            evidence=[f"{pid}: positive_ratio={ratio:.3f}, sample_n={n}" for pid, ratio, n in scored[:3]],
            risks_or_limitations=["Aspect-level sentiment extraction is not included in this build."],
        )
