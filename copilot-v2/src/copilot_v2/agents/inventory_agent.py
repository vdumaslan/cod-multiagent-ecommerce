from __future__ import annotations


def run_inventory(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    _ = retrieval_output
    return {
        "summary": (
            f"Inventory pass for '{user_query}': enforce 21-day stock coverage guardrails "
            "before demand-shaping promotions."
        ),
        "stockout_risk_level": "medium",
    }

