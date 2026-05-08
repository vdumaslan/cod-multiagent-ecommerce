"""Critic role: challenge the plan and surface risks."""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm import OllamaClient, OllamaChatResult, extract_json_object, validate_peer_review


def _fix_fact_claims(result: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace hallucinated pricing_source / price_missing values in Critic prose with ground truth."""
    pid_to_facts: dict[str, dict[str, str]] = {}
    for c in candidates:
        pid = str(c.get("product_id") or "").strip()
        if not pid:
            continue
        pr = c.get("pricing") or {}
        src = str(pr.get("source") or "fallback")
        pm = "true" if pr.get("price_missing", src != "cache") else "false"
        pid_to_facts[pid] = {"pricing_source": src, "price_missing": pm}

    def _fix(s: str) -> str:
        for pid, facts in pid_to_facts.items():
            if pid not in s:
                continue
            src = facts["pricing_source"]
            pm = facts["price_missing"]
            s = re.sub(r"pricing_source=['\"][\w]+['\"]", f"pricing_source='{src}'", s)
            s = re.sub(r"price_missing=(true|false)", f"price_missing={pm}", s, flags=re.IGNORECASE)
            # Sentence-level cleanup: when the product has a cache signal, remove phrases
            # that imply pricing is unavailable (leakage from few-shot fallback examples).
            if src == "cache":
                s = re.sub(
                    r";?\s*repricing without a cache signal is ungrounded\.?",
                    "; pricing signal is available — verify other risk signals before repricing.",
                    s, flags=re.IGNORECASE,
                )
                s = re.sub(
                    r"no pricing model data available\s*[—\-–]\s*cannot verify pricing direction[^.]*\.?",
                    "pricing model data available from cache; verify direction against sentiment and returns.",
                    s, flags=re.IGNORECASE,
                )
                s = re.sub(
                    r"cannot ground a reprice\s*[—\-–]\s*investigate instead\.?",
                    "pricing signal is available; investigate only if other risk signals are elevated.",
                    s, flags=re.IGNORECASE,
                )
                s = re.sub(
                    r"price_missing=true\s*\(fallback source\)[^;.]*[;.]?",
                    "price_missing=false (cache source available).",
                    s, flags=re.IGNORECASE,
                )
        return s

    for key in ("agreements", "disagreements", "suggested_changes"):
        result[key] = [_fix(c) for c in (result.get(key) or [])]
    return result


def _extract_pid(text: str) -> str | None:
    """Extract the first product_id-looking token from a free-text string."""
    m = re.search(r"\b([A-Z0-9]{8,20})\b", str(text or ""))
    return m.group(1) if m else None


def validate_critic_output(
    critic: dict[str, Any],
    advocate: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Deterministic post-processor that enforces structural consistency rules.

    Returns (cleaned_critic, list_of_warnings) so callers can log what was repaired.

    Rules applied:
    1. Remove a product_id from agreements if it also appears in disagreements.
       (The prompt says "don't do this", but 7B models still occasionally do.)
    2. Remove disagreement claims that falsely attribute an action to the Advocate
       if that product_id does not appear in the Advocate's proposed_actions at all.
       (Avoids the "Advocate proposed X for pid Y" when Y was never proposed.)
    3. Remove stockout-risk language from any product that is not stockout_risk in inventory.
    """
    warnings: list[str] = []

    # Build lookup structures
    advocate_pids: set[str] = {
        str((a or {}).get("product_id") or "").strip()
        for a in (advocate.get("proposed_actions") or [])
        if (a or {}).get("product_id")
    }
    candidate_stock: dict[str, str] = {
        str(c.get("product_id") or ""): str((c.get("inventory") or {}).get("stock_status") or "unknown")
        for c in candidates
        if c.get("product_id")
    }

    # Rule 1: no same-pid in both agreements and disagreements
    disagreement_pids: set[str] = set()
    for d in (critic.get("disagreements") or []):
        pid = _extract_pid(d)
        if pid:
            disagreement_pids.add(pid)

    cleaned_agreements = []
    for a in (critic.get("agreements") or []):
        pid = _extract_pid(a)
        if pid and pid in disagreement_pids:
            warnings.append(
                f"Removed agreement for {pid}: same product_id also in disagreements (contradicts itself)."
            )
        else:
            cleaned_agreements.append(a)
    critic["agreements"] = cleaned_agreements

    # Rule 2: remove disagreements that reference a pid not in Advocate's proposed_actions
    # Only applies when Advocate proposals are available (non-empty set).
    if advocate_pids:
        cleaned_disagreements = []
        for d in (critic.get("disagreements") or []):
            pid = _extract_pid(d)
            # "Advocate proposed X for pid Y" is only valid if Y was actually proposed.
            is_advocate_attribution = bool(
                re.search(r"\bAdvocate\s+proposed\b", str(d), re.IGNORECASE)
            )
            if is_advocate_attribution and pid and pid not in advocate_pids:
                warnings.append(
                    f"Removed disagreement for {pid}: falsely attributes an Advocate proposal "
                    f"(pid not in Advocate's proposed_actions)."
                )
            else:
                cleaned_disagreements.append(d)
        critic["disagreements"] = cleaned_disagreements

    # Rule 3: remove stockout-risk language for healthy/low_stock products
    for key in ("disagreements", "suggested_changes", "agreements"):
        cleaned = []
        for item in (critic.get(key) or []):
            pid = _extract_pid(item)
            stock = candidate_stock.get(pid or "", "unknown") if pid else "unknown"
            has_stockout_phrase = bool(re.search(r"\bstockout[_\s]risk\b", str(item), re.IGNORECASE))
            if has_stockout_phrase and stock not in ("stockout_risk", "unknown"):
                new_item = re.sub(
                    r"\bstockout[_\s]risk\b",
                    f"low_stock (actual status: {stock})",
                    str(item), flags=re.IGNORECASE,
                )
                warnings.append(
                    f"Replaced 'stockout_risk' language for {pid} (actual stock_status={stock})."
                )
                cleaned.append(new_item)
            else:
                cleaned.append(item)
        critic[key] = cleaned

    return critic, warnings


_SYSTEM = (
    "You are the Critic. Your role is to challenge the Advocate plan and surface risks. "
    "Focus on constraint violations, missing signals, and conflicts. Output STRICT JSON only."
)

_FEW_SHOT = (
    "FORMAT-ONLY example (do not use these product_ids — show grounded disagreements):\n"
    '{"agreements":['
    '"EXAMPLE_1: reprice at model_price_change_signal_pct=2.5% is consistent with cache signal and p_neg=0.04."'
    '],"disagreements":['
    '"EXAMPLE_2 (product_id=EXAMPLE_2): Advocate proposed reprice=+7.6% but large_delta=true and total_returns=6 — high delta on a high-return SKU increases risk; suggest investigate.",'
    '"EXAMPLE_3 (product_id=EXAMPLE_3): price_missing=true (fallback source); no pricing model data available — cannot verify pricing direction; suggest investigate."'
    '],"suggested_changes":['
    '"EXAMPLE_2: change action_type to investigate; set recommended_price_change_pct=0.0 until return cause is identified.",'
    '"EXAMPLE_3: change action_type to investigate; do not reprice when price_missing=true."'
    ']}\n'
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
                    "moderate_delta": (c.get("pricing", {}) or {}).get("moderate_delta", False),
                    "large_delta": (c.get("pricing", {}) or {}).get("large_delta", False),
                    "near_bound": (c.get("pricing", {}) or {}).get("near_bound", False),
                    "shrink_applied": (c.get("pricing", {}) or {}).get("shrink_applied", False),
                    "shrink_factor": (c.get("pricing", {}) or {}).get("shrink_factor", 1.0),
                },
                # Renamed: this is the PRICING MODEL's signal, not the Advocate's confirmed proposal.
                "model_price_change_signal_pct": c.get("recommended_price_change_pct"),
                "inventory_status": c.get("inventory", {}).get("stock_status"),
                "risk_flag": c.get("inventory", {}).get("risk_flag"),
                "available_to_sell": c.get("signals", {}).get("available_to_sell"),
                "total_units_sold": c.get("signals", {}).get("total_units_sold"),
                "total_returns": c.get("signals", {}).get("total_returns"),
                "return_rate": c.get("signals", {}).get("return_rate"),
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
                "- If Advocate matches the candidate model_price_change_signal_pct, say so (agreement) unless a risk/constraint argues otherwise.\n"
                "- Do NOT list the same product_id in both agreements and disagreements. Choose one or the other.\n"
                "Delta flags: moderate_delta=apply carefully; large_delta=human review recommended; "
                "large_delta alone does NOT require hold unless combined with p_neg>=0.30 (n_reviews>=10) or total_returns>=5.\n"
                "Returns: use return_rate when available (not raw count alone). "
                "total_returns>=5 AND return_rate>=0.05 = suggest investigate. "
                "return_rate 0.03–0.05 = note as moderate risk, do not hard-block. "
                "High-volume products may have low rates despite high counts — always cite the rate.\n"
                "Inventory language: only say 'stockout risk' if inventory_status='stockout_risk'; "
                "only say 'low stock' if inventory_status='low_stock' or risk_flag=true; "
                "for healthy inventory state it is healthy.\n"
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

    validated = validate_peer_review(obj)
    fact_fixed = _fix_fact_claims(validated, payload.get("candidates", []))
    # Deterministic consistency checks: remove same-product agreements+disagreements,
    # false Advocate attributions, and wrong inventory-language.
    cleaned, repair_warnings = validate_critic_output(
        fact_fixed,
        advocate=payload.get("advocate") or {},
        candidates=payload.get("candidates") or [],
    )
    if repair_warnings:
        raw["critic_repairs"] = repair_warnings
    return cleaned, raw
