"""Advocate role: argue for the strongest plan, and revise based on critic + human feedback."""
from __future__ import annotations

import json
from typing import Any

from app.llm import OllamaClient, OllamaChatResult, extract_json_object, validate_specialist_proposal, SchemaError


_SYSTEM = (
    "You are the Advocate. Your role is to argue for the strongest grounded action plan. "
    "You must be concise, grounded in the provided candidate facts only, and output STRICT JSON only."
)

_SYSTEM_REVISION = (
    "You are the Advocate. You are revising your previous proposal based on Critic feedback "
    "and optional human input. Refine your plan — do not simply repeat the original. "
    "Output STRICT JSON only. "
    "Adversarial requirement: you MUST change at least one proposed action (action_type or recommended_price_change_pct) "
    "relative to round1_advocate unless the Critic had zero meaningful disagreements; if you keep an action unchanged, "
    "explicitly justify it in key_claims by citing a provided signal/constraint."
)

_FEW_SHOT = (
    "FORMAT-ONLY examples (do not use these product_ids):\n"
    '{"proposed_actions":['
    '{"product_id":"EXAMPLE_1","action_type":"reprice","recommended_price_change_pct":2.5},'
    '{"product_id":"EXAMPLE_2","action_type":"hold","recommended_price_change_pct":0.0},'
    '{"product_id":"EXAMPLE_3","action_type":"promote","recommended_price_change_pct":0.0}'
    '],"key_claims":["..."],"concerns":["..."]}\n'
)


def build_messages(
    *,
    payload: dict[str, Any],
    top_k: int,
    prompt_style: str = "few_shot_json",
    prompt_version: str = "v1",
    is_revision: bool = False,
) -> list[dict[str, str]]:
    style = str(prompt_style or "few_shot_json").strip()
    few_shot = _FEW_SHOT if style == "few_shot_json" else ""
    cot = (
        "Before answering, think step-by-step privately. Do NOT output your reasoning.\n"
        if style == "cot_hidden"
        else ""
    )
    system = _SYSTEM_REVISION if is_revision else _SYSTEM

    if is_revision:
        task = (
            "Task: revise your proposal based on the Critic's challenges and human input (if provided).\n"
            "Address the disagreements. Do not repeat the same plan unchanged.\n"
        )
        # Surface critic feedback and human input prominently
        critic_note = ""
        if payload.get("round1_critic"):
            critic_note = f"CRITIC_FEEDBACK:\n{payload['round1_critic']}\n\n"
        human_note = ""
        if payload.get("human_feedback"):
            human_note = f"HUMAN_INPUT:\n{payload['human_feedback']}\n\n"
        context = critic_note + human_note
    else:
        task = "Task: argue for the strongest grounded action plan.\n"
        context = ""

    slim_candidates = [
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
            "recommended_price_change_pct": c.get("recommended_price_change_pct", 0.0),
            "sentiment": {k: c.get("sentiment", {}).get(k) for k in ("p_pos", "p_neu", "p_neg", "n_reviews")},
            "inventory_status": c.get("inventory", {}).get("stock_status"),
            "risk_flag": c.get("inventory", {}).get("risk_flag"),
            "available_to_sell": c.get("signals", {}).get("available_to_sell"),
            "mean_daily_revenue": c.get("signals", {}).get("mean_daily_revenue"),
            "total_returns": c.get("signals", {}).get("total_returns"),
            "retrieval_score": c.get("evidence", {}).get("retrieval_score"),
        }
        for c in payload.get("candidates", [])
    ]
    slim_payload = {
        "goal": payload.get("goal"),
        "objective": (payload.get("constraints") or {}).get("objective", "revenue"),
        "constraints": payload.get("constraints"),
        "candidates": slim_candidates,
        "baseline_actions": [
            {"product_id": a.get("product_id"), "recommended_price_change_pct": a.get("recommended_price_change_pct")}
            for a in payload.get("baseline_actions", [])
        ],
    }
    if payload.get("round1_advocate"):
        slim_payload["round1_advocate"] = payload["round1_advocate"]
    if payload.get("round1_critic"):
        slim_payload["round1_critic"] = payload["round1_critic"]
    if payload.get("human_feedback"):
        slim_payload["human_feedback"] = payload["human_feedback"]

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"prompt_style={style} prompt_version={prompt_version}\n"
                + task
                + cot
                + few_shot
                + context
                + "Return STRICT JSON only. No markdown, no explanation, no extra text.\n"
                f"Keys: proposed_actions (top {top_k}), key_claims (<=5 strings), concerns (<=5 strings).\n"
                "Each proposed_action must have: product_id (string), action_type (string), recommended_price_change_pct (number).\n"
                "Allowed action_type values: reprice, hold, promote, investigate, restock.\n"
                "Grounding rules:\n"
                "- Use the candidate's recommended_price_change_pct as your starting point for any reprice action.\n"
                "- If pricing_source is 'fallback', prefer action_type='investigate' and set recommended_price_change_pct=0.0.\n"
                "- Do not flip the sign of the candidate recommended_price_change_pct unless you cite a specific risk/constraint signal.\n"
                "- For non-reprice actions, set recommended_price_change_pct=0.0.\n"
                "Use only product_ids from candidates below.\n"
                f"INPUT_JSON:\n{json.dumps(slim_payload)}"
            ),
        },
    ]


def _fallback(candidates: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    return {
        "proposed_actions": [
            {
                "product_id": str(c.get("product_id")),
                "action_type": "reprice",
                "recommended_price_change_pct": float(c.get("recommended_price_change_pct") or 0.0),
            }
            for c in candidates[:top_k]
            if c.get("product_id")
        ],
        "key_claims": ["Fallback: advocate JSON parse/validation failed."],
        "concerns": [],
    }


def run(
    ollama: OllamaClient,
    *,
    model: str,
    payload: dict[str, Any],
    top_k: int,
    prompt_style: str,
    prompt_version: str,
    seed: int = 42,
    temperature: float = 0.2,
    num_predict: int = 800,
    is_revision: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (validated_advocate, raw_trace)."""
    allowed = {str(c.get("product_id")) for c in payload.get("candidates", []) if c.get("product_id")}
    messages = build_messages(
        payload=payload,
        top_k=top_k,
        prompt_style=prompt_style,
        prompt_version=prompt_version,
        is_revision=is_revision,
    )
    trace_key = "advocate_revision" if is_revision else "advocate"
    raw: dict[str, Any] = {}

    res: OllamaChatResult = ollama.chat(
        model=model, messages=messages, temperature=temperature, num_predict=num_predict, seed=seed
    )
    raw[trace_key] = {"elapsed_s": res.elapsed_s, "content": res.content[:3000]}

    obj: dict[str, Any] | None = None
    try:
        obj = extract_json_object(res.content)
    except Exception:
        pass

    for i, hint in enumerate([
        "Return ONE valid JSON object only.",
        '{"proposed_actions":[],"key_claims":[],"concerns":[]}',
    ]):
        if obj is not None:
            break
        res2 = ollama.chat(
            model=model,
            messages=messages + [{"role": "user", "content": hint}],
            temperature=0.0, num_predict=num_predict, seed=seed + 10 + i,
        )
        raw[f"{trace_key}_retry_{i}"] = {"elapsed_s": res2.elapsed_s, "content": res2.content[:3000]}
        try:
            obj = extract_json_object(res2.content)
        except Exception:
            pass

    if obj is None:
        raw[f"{trace_key}_fallback"] = True
        return _fallback(payload.get("candidates", []), top_k), raw

    try:
        return validate_specialist_proposal(obj, allowed=allowed, top_k=top_k), raw
    except SchemaError:
        raw[f"{trace_key}_fallback"] = True
        return _fallback(payload.get("candidates", []), top_k), raw
