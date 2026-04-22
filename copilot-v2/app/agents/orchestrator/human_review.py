"""Human review step: add context for another round, or proceed to judge."""
from __future__ import annotations

from typing import Any, Literal

HumanReviewMode = Literal["skip", "second_round", "second_round_with_feedback"]


def decide(
    debate_state: dict[str, Any],
    *,
    mode: HumanReviewMode = "skip",
    human_feedback: str | None = None,
) -> dict[str, Any]:
    """
    Evaluates the human review decision and returns an enriched state dict.

    Modes:
      skip                        — proceed directly to judge, no second round
      second_round                — re-run advocate + critic with same inputs, then judge
      second_round_with_feedback  — inject human_feedback into payload, re-run advocate + critic, then judge

    Returns the debate_state with two added keys:
      do_second_round: bool
      human_feedback:  str | None  (only set for second_round_with_feedback)
    """
    do_second_round = mode in ("second_round", "second_round_with_feedback")
    injected_feedback = human_feedback if mode == "second_round_with_feedback" else None

    return {
        **debate_state,
        "do_second_round": do_second_round,
        "human_feedback": injected_feedback,
    }
