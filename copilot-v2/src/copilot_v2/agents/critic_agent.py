from __future__ import annotations


def run_critic(
    *,
    user_query: str,
    retrieval_output: dict[str, object],
    sentiment_output: dict[str, object],
    pricing_output: dict[str, object],
    inventory_output: dict[str, object],
    advocate_output: str,
    previous_advocate_output: str = "",
    previous_critic_output: str = "",
    added_context: str = "",
    round_number: int = 1,
) -> dict[str, str]:
    memory_parts: list[str] = []
    if previous_advocate_output:
        memory_parts.append(f"prior advocate: {previous_advocate_output}")
    if previous_critic_output:
        memory_parts.append(f"prior critic: {previous_critic_output}")
    if added_context:
        memory_parts.append(f"user context: {added_context}")
    memory_text = " | ".join(memory_parts) if memory_parts else "no prior round memory"

    message = (
        f"Round {round_number} critic review for '{user_query}'. "
        f"Using same base outputs plus advocate response. "
        f"Retrieval={retrieval_output.get('summary', '')}; "
        f"Sentiment={sentiment_output.get('summary', '')}; "
        f"Pricing={pricing_output.get('summary', '')}; "
        f"Inventory={inventory_output.get('summary', '')}; "
        f"Advocate={advocate_output}. "
        f"Memory={memory_text}. "
        "New concerns: campaign concentration, promo fatigue, and regional stock volatility."
    )
    return {"message": message}

