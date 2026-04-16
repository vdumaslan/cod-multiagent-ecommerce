from __future__ import annotations


def run_sentiment(*, user_query: str, retrieval_output: dict[str, object]) -> dict[str, object]:
    retrieval_summary = str(retrieval_output.get("summary", "retrieval summary unavailable"))
    return {
        "summary": (
            f"Sentiment pass for '{user_query}': customers react positively to bundles, "
            "but trust drops with aggressive discount cadence."
        ),
        "grounding": retrieval_summary,
    }

