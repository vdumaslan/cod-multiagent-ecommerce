from __future__ import annotations


def run_pricing(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    sample = retrieval_output.get("selected_sample", {})
    baseline_actions = sample.get("baseline_actions", []) if isinstance(sample, dict) else []
    avg_change = 0.0
    if baseline_actions:
        deltas = [
            float(a.get("recommended_price_change_pct", 0.0))
            for a in baseline_actions
            if isinstance(a, dict)
        ]
        avg_change = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "summary": (
            f"Pricing pass for '{user_query}': replay baseline average price delta is {avg_change:.2f}%, "
            "so plans should focus on selective repricing over broad discounting."
        ),
        "suggested_price_change_pct": avg_change,
    }

