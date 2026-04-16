from __future__ import annotations


def run_sentiment(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    retrieval_summary = str(retrieval_output.get("summary", "retrieval summary unavailable"))
    sample = retrieval_output.get("selected_sample", {})
    candidates = sample.get("candidates", []) if isinstance(sample, dict) else []
    avg_returns = 0.0
    if candidates:
        returns = [
            float(c.get("signals", {}).get("total_returns", 0.0))
            for c in candidates
            if isinstance(c, dict)
        ]
        avg_returns = sum(returns) / len(returns) if returns else 0.0
    return {
        "summary": (
            f"Sentiment pass for '{user_query}': replay sample indicates average returns "
            f"of {avg_returns:.1f} across candidate products, suggesting moderate trust sensitivity."
        ),
        "grounding": retrieval_summary,
    }

