from __future__ import annotations


def run_retrieval(*, user_query: str) -> dict[str, object]:
    query = user_query.strip()
    return {
        "summary": f"Retrieved opportunities and constraints for '{query}'.",
        "top_findings": [
            "High-repeat segments respond well to curated bundles.",
            "Top paid channels show concentration risk in two campaigns.",
            "Stock-sensitive categories need promotion guardrails.",
        ],
    }

