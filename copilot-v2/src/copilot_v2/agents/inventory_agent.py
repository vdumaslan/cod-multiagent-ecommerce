from __future__ import annotations


def run_inventory(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    sample = retrieval_output.get("selected_sample", {})
    candidates = sample.get("candidates", []) if isinstance(sample, dict) else []
    risk_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        inventory = candidate.get("inventory", {})
        if isinstance(inventory, dict) and bool(inventory.get("risk_flag")):
            risk_count += 1
    risk_level = "low" if risk_count == 0 else ("medium" if risk_count < 2 else "high")
    return {
        "summary": (
            f"Inventory pass for '{user_query}': replay sample shows {risk_count} high-risk candidate(s), "
            "so keep stock guardrails before demand-shaping promotions."
        ),
        "stockout_risk_level": risk_level,
    }

