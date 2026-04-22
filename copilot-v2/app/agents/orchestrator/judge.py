"""Judge role: synthesize debate into ranked plans."""
from __future__ import annotations

from typing import Any

from app.llm import OllamaClient, OllamaChatResult, extract_json_object, validate_ranked_actions, SchemaError


_SYSTEM = (
    "You are the Judge. Synthesize the Advocate and Critic into a final ranked action plan. "
    "Output STRICT JSON only with key ranked_actions. "
    "Use only candidate product_ids. Respect max_abs_price_change_pct."
)

_FEW_SHOT = (
    "FORMAT-ONLY example:\n"
    '{"ranked_actions":[{"product_id":"EXAMPLE_1","action_type":"reprice","recommended_price_change_pct":0.0,'
    '"rationale_bullets":["...","..."],"risk_bullets":["..."]}]}\n'
)


def build_messages(
    *,
    payload: dict[str, Any],
    top_k: int,
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
    extra = (
        "Write rationale_bullets as short, grounded evidence statements (2–4 bullets).\n"
        "Write risk_bullets as constraint/risk checks (1–3 bullets).\n"
        if style == "structured_rationale"
        else ""
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"prompt_style={style} prompt_version={prompt_version}\n" + cot + few_shot + extra + "Return JSON only.\n"},
        {"role": "user", "content": f"INPUT_JSON:\n{payload}"},
    ]


def _fallback_from_baseline(
    baseline_actions: list[dict[str, Any]], allowed: set[str], top_k: int
) -> tuple[list[dict[str, Any]], bool]:
    out = []
    for a in baseline_actions[:top_k]:
        pid = str((a or {}).get("product_id") or "").strip()
        if not pid or pid not in allowed:
            continue
        out.append({
            "product_id": pid,
            "action_type": str(a.get("action_type") or "reprice").lower() or "reprice",
            "recommended_price_change_pct": float(a.get("recommended_price_change_pct") or 0.0),
            "rationale_bullets": ["Fallback: judge output did not validate; using baseline plan."],
            "risk_bullets": ["LLM JSON validation failed; baseline preserved."],
        })
        if len(out) >= top_k:
            break
    return out, True


def run(
    ollama: OllamaClient,
    *,
    model: str,
    payload: dict[str, Any],
    baseline_actions: list[dict[str, Any]],
    top_k: int,
    prompt_style: str,
    prompt_version: str,
    seed: int = 63,
    temperature: float = 0.2,
    num_predict: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Returns (final_actions, raw_trace, judge_fallback)."""
    allowed = {str(c.get("product_id")) for c in payload.get("candidates", []) if c.get("product_id")}
    allowed_hint = ", ".join(sorted(allowed)[:24]) + (", ..." if len(allowed) > 24 else "")
    messages = build_messages(payload=payload, top_k=top_k, prompt_style=prompt_style, prompt_version=prompt_version)
    raw: dict[str, Any] = {}

    res: OllamaChatResult = ollama.chat(
        model=model, messages=messages, temperature=temperature, num_predict=num_predict, seed=seed
    )
    raw["judge"] = {"elapsed_s": res.elapsed_s, "content": res.content[:3000]}

    final_actions: list[dict[str, Any]] | None = None
    try:
        obj = extract_json_object(res.content)
        final_actions = validate_ranked_actions(obj, allowed=allowed, top_k=top_k)
    except Exception as e:
        raw["judge_parse_error"] = str(e)[:300]

    if final_actions is None:
        repair = (
            f"Return JSON with key ranked_actions (list, max {top_k} items). "
            f"Each item: product_id (one of: {allowed_hint}), action_type, recommended_price_change_pct, "
            "rationale_bullets (2-4 strings), risk_bullets (1-3 strings). No markdown."
        )
        res2 = ollama.chat(
            model=model, messages=messages + [{"role": "user", "content": repair}],
            temperature=0.0, num_predict=num_predict, seed=seed + 31,
        )
        raw["judge_retry"] = {"elapsed_s": res2.elapsed_s, "content": res2.content[:3000]}
        try:
            obj2 = extract_json_object(res2.content)
            final_actions = validate_ranked_actions(obj2, allowed=allowed, top_k=top_k)
        except Exception as e2:
            raw["judge_retry_error"] = str(e2)[:300]

    if final_actions is None:
        raw["used_baseline_fallback"] = True
        final_actions, fallback = _fallback_from_baseline(baseline_actions, allowed, top_k)
        return final_actions, raw, fallback

    return final_actions, raw, False
