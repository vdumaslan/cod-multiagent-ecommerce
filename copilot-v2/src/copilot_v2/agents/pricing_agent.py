from __future__ import annotations


def run_pricing(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    _ = retrieval_output
    return {
        "summary": (
            f"Pricing pass for '{user_query}': use tiered discounts with category-level margin floors "
            "and pause rules for weak contribution SKUs."
        ),
        "suggested_price_change_pct": 4.5,
    }

