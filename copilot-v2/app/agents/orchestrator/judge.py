"""Judge role: synthesize debate into ranked plans."""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm import OllamaClient, OllamaChatResult, extract_json_object, validate_ranked_actions, SchemaError


_SYSTEM = (
    "You are the Judge. Synthesize the Advocate and Critic into a final ranked action plan. "
    "Output STRICT JSON only with key ranked_actions. "
    "Use only candidate product_ids. Respect max_abs_price_change_pct. "
    "Allowed action_type values: reprice, hold, promote, investigate, restock. "
    "Debate linkage rule:\n"
    "- Default to the Advocate's latest proposed_actions for each product_id. "
    "Only deviate if the Critic raised a concrete objection for that product_id OR a hard rule applies (e.g., low_stock guardrail). "
    "If you deviate, explicitly cite the Critic objection (or hard rule) in rationale_bullets.\n"
    "IMPORTANT decision rules:\n"
    "- If a candidate has inventory.stock_status in {low_stock, stockout_risk} OR risk_flag=true, default to action_type='hold' (or 'restock' if stockout_risk). "
    "Do NOT recommend raising price or running promotions for low-stock items.\n"
    "- Only override suggested_action if you can cite a specific provided signal in rationale_bullets.\n"
    "- Use recommended_price_change_pct only for action_type='reprice'. For other action types set it to 0.0."
    "\n"
    "Objective handling (from constraints.objective):\n"
    "- revenue: favor actions that plausibly increase sales volume or price where safe.\n"
    "- profit: be conservative on discounting; prefer hold/investigate unless signals support repricing.\n"
    "- clear_inventory: for overstocked items, favor promote or reprice down when safe.\n"
    "- reduce_returns: for high returns/negative sentiment, favor investigate/promote fixes over repricing.\n"
    "- avoid_stockouts: for low_stock/stockout_risk, favor hold/restock (already required above).\n"
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
        {"role": "user", "content": f"INPUT_JSON:\n{json.dumps(payload)}"},
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

    # Build advocate proposal map (for deterministic debate linkage post-processing).
    adv_map: dict[str, dict[str, Any]] = {}
    for a in ((payload.get("advocate") or {}).get("proposed_actions") or []):
        pid = str((a or {}).get("product_id") or "").strip()
        if not pid:
            continue
        adv_map[pid] = {
            "action_type": str((a or {}).get("action_type") or "reprice").strip().lower() or "reprice",
            "recommended_price_change_pct": float((a or {}).get("recommended_price_change_pct") or 0.0),
        }

    # Parse critic disagreements to find which product_ids are explicitly disputed.
    critic_texts: list[str] = []
    crit = payload.get("critic") or {}
    if isinstance(crit, dict):
        for k in ("disagreements", "suggested_changes"):
            v = crit.get(k)
            if isinstance(v, list):
                critic_texts.extend([str(x) for x in v if x is not None])
            elif v is not None:
                critic_texts.append(str(v))
    critic_disputed: set[str] = set()
    for t in critic_texts:
        # Accept formats like "{'product_id': 'B01...'}" or "product_id=B01..."
        m = re.search(r"product_id[^A-Za-z0-9]*([A-Za-z0-9]{8,20})", t)
        if m:
            critic_disputed.add(m.group(1))

    # Post-process for inventory safety: do not let the judge suggest price/promo actions
    # for low-stock / stockout candidates even if the LLM ignores instructions.
    by_pid = {
        str(c.get("product_id")): (c or {})
        for c in (payload.get("candidates") or [])
        if (c or {}).get("product_id")
    }
    for a in final_actions:
        pid = str(a.get("product_id") or "")
        cand = by_pid.get(pid, {})
        inv = (cand.get("inventory") or {})
        stock_status = str(inv.get("stock_status") or "unknown").lower()
        risk_flag = bool(inv.get("risk_flag", False))
        suggested = str(cand.get("suggested_action") or "").strip().lower()

        if stock_status == "stockout_risk" or (stock_status in {"low_stock"} or risk_flag):
            # Default to hold (or restock for stockout_risk) regardless of LLM choice.
            a["action_type"] = "restock" if stock_status == "stockout_risk" else "hold"
            a["recommended_price_change_pct"] = 0.0
        elif suggested in {"hold", "restock", "promote", "investigate"} and a.get("action_type") == "reprice":
            # If the playbook strongly suggests a non-reprice action, allow it to override reprice.
            a["action_type"] = suggested
            a["recommended_price_change_pct"] = 0.0

    # Deterministic debate linkage: keep the Advocate's final proposal unless the Critic explicitly disputed that SKU
    # (or a hard inventory guardrail applied above).
    linked = 0
    for a in final_actions:
        pid = str(a.get("product_id") or "").strip()
        if not pid or pid not in adv_map:
            continue
        # If the critic disputed this pid, allow judge independence.
        if pid in critic_disputed:
            continue
        cand = by_pid.get(pid, {})
        inv = (cand.get("inventory") or {})
        stock_status = str(inv.get("stock_status") or "unknown").lower()
        risk_flag = bool(inv.get("risk_flag", False))
        # If guardrail applies, do not override.
        if stock_status in {"low_stock", "stockout_risk"} or risk_flag:
            continue
        a["action_type"] = adv_map[pid]["action_type"]
        a["recommended_price_change_pct"] = float(adv_map[pid]["recommended_price_change_pct"] or 0.0)
        linked += 1
    raw["debate_linkage"] = {"linked_to_advocate": linked, "critic_disputed_n": len(critic_disputed)}

    # Also align product selection: by default, rank the same product_ids as the Advocate proposed (top_k),
    # unless a hard guardrail changed the action for that pid.
    adv_pids_in_order = [pid for pid in [str((a or {}).get("product_id") or "").strip()
                        for a in ((payload.get("advocate") or {}).get("proposed_actions") or [])] if pid]
    if adv_pids_in_order:
        by_judge_pid = {str(a.get("product_id") or "").strip(): a for a in final_actions if a.get("product_id")}
        linked_actions: list[dict[str, Any]] = []
        for pid in adv_pids_in_order:
            if pid not in allowed:
                continue
            a = by_judge_pid.get(pid)
            if a is None:
                # If judge omitted this pid, synthesize from advocate proposal (still passes schema).
                # Leave rationale_bullets empty so downstream can fill deterministic, signal-based bullets.
                a = {
                    "product_id": pid,
                    "action_type": adv_map.get(pid, {}).get("action_type", "reprice"),
                    "recommended_price_change_pct": float(adv_map.get(pid, {}).get("recommended_price_change_pct") or 0.0),
                    "rationale_bullets": [],
                    "risk_bullets": [],
                }
            linked_actions.append(a)
            if len(linked_actions) >= top_k:
                break
        if linked_actions:
            final_actions = linked_actions

    return final_actions, raw, False
