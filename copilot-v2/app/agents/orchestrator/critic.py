"""Critic role: challenge the plan and surface risks."""
from __future__ import annotations

import json
from typing import Any

from app.llm import OllamaClient, OllamaChatResult, extract_json_object, validate_peer_review


_SYSTEM = (
    "You are the Critic. Your role is to challenge the Advocate plan and surface risks. "
    "Focus on constraint violations, missing signals, and conflicts. Output STRICT JSON only."
)

_FEW_SHOT = (
    "FORMAT-ONLY example:\n"
    '{"agreements":["..."],"disagreements":["..."],"suggested_changes":["..."]}\n'
)


def build_messages(
    *,
    payload: dict[str, Any],
    prompt_style: str = "few_shot_json",
    prompt_version: str = "v1",
) -> list[dict[str, str]]:
    style = str(prompt_style or "few_shot_json").strip()
    few_shot = _FEW_SHOT if style == "few_shot_json" else ""
    cot = (
        "Before answering, think step-by-step privately. Do NOT output your reasoning.\n"
        if style == "cot_hidden"
        else ""
    )
    slim_payload = {
        "goal": payload.get("goal"),
        "constraints": payload.get("constraints"),
        "advocate": payload.get("advocate"),
        "baseline_actions": [
            {"product_id": a.get("product_id"), "recommended_price_change_pct": a.get("recommended_price_change_pct")}
            for a in payload.get("baseline_actions", [])
        ],
        "candidates": [
            {
                "product_id": c.get("product_id"),
                "suggested_action": c.get("suggested_action"),
                "pricing_source": c.get("pricing", {}).get("source"),
                "pricing_flags": {
                    "large_delta": (c.get("pricing", {}) or {}).get("large_delta", False),
                    "near_bound": (c.get("pricing", {}) or {}).get("near_bound", False),
                    "shrink_applied": (c.get("pricing", {}) or {}).get("shrink_applied", False),
                    "shrink_factor": (c.get("pricing", {}) or {}).get("shrink_factor", 1.0),
                },
                "recommended_price_change_pct": c.get("recommended_price_change_pct"),
                "inventory_status": c.get("inventory", {}).get("stock_status"),
                "risk_flag": c.get("inventory", {}).get("risk_flag"),
                "available_to_sell": c.get("signals", {}).get("available_to_sell"),
                "total_returns": c.get("signals", {}).get("total_returns"),
            }
            for c in payload.get("candidates", [])
        ],
    }
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"prompt_style={style} prompt_version={prompt_version}\n"
                "Task: challenge the Advocate plan using the same grounded inputs.\n"
                "Return STRICT JSON only. No markdown, no extra text.\n"
                "Keys: agreements (<=6 strings), disagreements (<=6 strings), suggested_changes (<=6 strings).\n"
                "Focus on risks, constraint violations, and conflicts between signals.\n"
                "Adversarial requirements:\n"
                "- Each disagreement MUST reference a specific product_id and a concrete signal/constraint (e.g., pricing_source=fallback, p_neg, total_returns, inventory_status, max_abs_price_change_pct).\n"
                "- At least 2 disagreements should propose an alternative with a concrete change (action_type or price_change) for that product.\n"
                "- For pricing_source=fallback, strongly prefer suggesting action_type='investigate' with 0.0% instead of repricing.\n"
                "- If Advocate matches the candidate recommended_price_change_pct, say so (agreement) unless a risk/constraint argues otherwise.\n"
                + cot
                + few_shot
                + f"INPUT_JSON:\n{json.dumps(slim_payload)}"
            ),
        },
    ]


def run(
    ollama: OllamaClient,
    *,
    model: str,
    payload: dict[str, Any],
    prompt_style: str,
    prompt_version: str,
    seed: int = 49,
    temperature: float = 0.2,
    num_predict: int = 800,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (validated_critic, raw_trace)."""
    messages = build_messages(payload=payload, prompt_style=prompt_style, prompt_version=prompt_version)
    raw: dict[str, Any] = {}

    res: OllamaChatResult = ollama.chat(
        model=model, messages=messages, temperature=temperature, num_predict=num_predict, seed=seed
    )
    raw["critic"] = {"elapsed_s": res.elapsed_s, "content": res.content[:3000]}

    obj: dict[str, Any] | None = None
    try:
        obj = extract_json_object(res.content)
    except Exception:
        pass

    if obj is None:
        res2 = ollama.chat(
            model=model,
            messages=messages + [{"role": "user", "content": "Return ONE valid JSON object only. Keys: agreements, disagreements, suggested_changes."}],
            temperature=0.0, num_predict=num_predict, seed=seed + 10,
        )
        raw["critic_retry"] = {"elapsed_s": res2.elapsed_s, "content": res2.content[:3000]}
        try:
            obj = extract_json_object(res2.content)
        except Exception:
            pass

    if obj is None:
        raw["critic_fallback"] = True
        return validate_peer_review({}), raw

    return validate_peer_review(obj), raw
