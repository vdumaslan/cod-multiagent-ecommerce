from __future__ import annotations

from collections import Counter

from .contracts import AgentOutput, DebateResult
from .llm_runtime import LLMRuntime


class OrchestratorAgent:
    def __init__(self, llm_model_id: str | None = None, llm_fallback_model_id: str | None = None) -> None:
        self.reasoner = (
            LLMRuntime(model_id=llm_model_id, fallback_model_id=llm_fallback_model_id) if llm_model_id else None
        )

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
        rationale = "Consensus derived from Discovery/Sentiment/Ranking/Pricing arguments."

        if self.reasoner is not None:
            trace_lines = []
            for trace in traces:
                top_item = trace.recommended_items[0] if trace.recommended_items else "none"
                trace_lines.append(
                    f"- {trace.agent_name}: top_item={top_item}, confidence={trace.confidence:.3f}, claim={trace.claim}"
                )
            llm_text = self.reasoner.generate(
                system_prompt=(
                    "You are an orchestration agent. Summarize multi-agent evidence for a seller. "
                    "Keep it to 2 short sentences and include one caveat."
                ),
                user_prompt=(
                    f"Winner by vote: {winner}\n"
                    f"Runner-up: {runner_up}\n"
                    f"Uncertainty: {uncertainty}\n"
                    "Agent traces:\n"
                    + "\n".join(trace_lines)
                ),
                max_new_tokens=140,
                temperature=0.1,
            )
            if llm_text:
                rationale = llm_text.strip()

        return DebateResult(
            winner=winner,
            runner_up=runner_up,
            rationale=rationale,
            uncertainty=uncertainty,
            traces=traces,
        )
