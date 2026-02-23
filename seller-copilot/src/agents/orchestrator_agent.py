from __future__ import annotations

from collections import Counter

from .contracts import AgentOutput, DebateResult


class OrchestratorAgent:
    def synthesize(self, traces: list[AgentOutput]) -> DebateResult:
        # Majority vote on top recommendation with confidence tie-break.
        top_votes = [t.recommended_items[0] for t in traces if t.recommended_items]
        if not top_votes:
            return DebateResult(
                winner="",
                runner_up=None,
                rationale="No valid recommendations from debater agents.",
                uncertainty=1.0,
                traces=traces,
            )

        counts = Counter(top_votes).most_common()
        winner = counts[0][0]
        runner_up = counts[1][0] if len(counts) > 1 else None
        avg_conf = sum(t.confidence for t in traces) / max(1, len(traces))
        uncertainty = round(1.0 - avg_conf, 4)

        return DebateResult(
            winner=winner,
            runner_up=runner_up,
            rationale="Consensus derived from Sentiment/Ranking/Pricing arguments.",
            uncertainty=uncertainty,
            traces=traces,
        )

