from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RubricScores:
    constraint_respect: float
    grounded_ids: float
    conflict_detection: float
    decision_justification: float
    json_valid: float


def _has_any_text(xs: Any) -> bool:
    if not isinstance(xs, list):
        return False
    return any(str(x).strip() for x in xs)


def score_debate_output(
    *,
    final_actions: list[dict[str, Any]] | None,
    allowed_product_ids: set[str],
    max_abs_price_change_pct: float = 10.0,
    candidates_by_pid: dict[str, dict[str, Any]] | None = None,
    critic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministic rubric scorer (no LLM).

    Intended to compare debate model combinations fairly when replaying the same specialist inputs.
    """
    if not final_actions or not isinstance(final_actions, list):
        return {**RubricScores(0, 0, 0, 0, 0).__dict__, "notes": ["missing_final_actions"]}

    # JSON validity: if we got here, it parsed and validated enough to be a list.
    json_valid = 1.0

    # Grounded IDs + constraint respect
    grounded_ok = 0
    constraint_ok = 0
    for a in final_actions:
        if not isinstance(a, dict):
            continue
        pid = str(a.get("product_id") or "").strip()
        if pid and pid in allowed_product_ids:
            grounded_ok += 1
        try:
            rec = float(a.get("recommended_price_change_pct") or 0.0)
        except Exception:
            rec = 0.0
        if abs(rec) <= float(max_abs_price_change_pct) + 1e-6:
            constraint_ok += 1

    denom = max(1, len(final_actions))
    grounded_ids = grounded_ok / denom
    constraint_respect = constraint_ok / denom

    # Decision justification: presence of rationale + risk bullets.
    just_ok = 0
    for a in final_actions:
        if not isinstance(a, dict):
            continue
        if _has_any_text(a.get("rationale_bullets")) and _has_any_text(a.get("risk_bullets")):
            just_ok += 1
    decision_justification = just_ok / denom

    # Conflict detection: credit if critic raises at least one disagreement/suggested change mentioning inventory conflict keywords
    # OR if judge risk bullets mention inventory risks.
    conflict_score = 0.0
    conflict_keywords = ("stock", "inventory", "overstock", "low_stock", "stockout", "returns")
    if critic and isinstance(critic, dict):
        text = " ".join([str(x) for x in (critic.get("disagreements") or []) + (critic.get("suggested_changes") or [])]).lower()
        if any(k in text for k in conflict_keywords):
            conflict_score = 1.0
    if conflict_score < 1.0:
        for a in final_actions:
            rb = " ".join([str(x) for x in (a.get("risk_bullets") or [])]).lower()
            if any(k in rb for k in conflict_keywords):
                conflict_score = 1.0
                break

    out = RubricScores(
        constraint_respect=float(constraint_respect),
        grounded_ids=float(grounded_ids),
        conflict_detection=float(conflict_score),
        decision_justification=float(decision_justification),
        json_valid=float(json_valid),
    ).__dict__
    return out

