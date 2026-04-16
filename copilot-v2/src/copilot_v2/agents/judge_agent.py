from __future__ import annotations


def run_judge(
    *,
    user_query: str,
    retrieval_output: dict[str, object],
    sentiment_output: dict[str, object],
    pricing_output: dict[str, object],
    inventory_output: dict[str, object],
    latest_advocate_output: str,
    latest_critic_output: str,
    added_context_history: list[str] | None = None,
) -> dict[str, str]:
    context_history = ", ".join(added_context_history or []) or "none"
    message = (
        f"Judge synthesis for '{user_query}'. "
        f"Retrieved={retrieval_output.get('summary', '')}; "
        f"Sentiment={sentiment_output.get('summary', '')}; "
        f"Pricing={pricing_output.get('summary', '')}; "
        f"Inventory={inventory_output.get('summary', '')}. "
        f"Latest advocate={latest_advocate_output}. Latest critic={latest_critic_output}. "
        f"Human context history={context_history}. "
        "Final direction: balanced growth plan with explicit margin and inventory controls."
    )
    return {"message": message}

