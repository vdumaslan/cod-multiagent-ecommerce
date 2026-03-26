from __future__ import annotations

import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODELS: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


def _get_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_name not in _MODELS:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        # Some model repos ship sampling-only defaults that trigger warnings
        # when we run deterministic generation (do_sample=False).
        if getattr(mdl, "generation_config", None) is not None:
            mdl.generation_config.temperature = None
            mdl.generation_config.top_p = None
            mdl.generation_config.top_k = None
        mdl.eval()
        _MODELS[model_name] = (mdl, tok)
    return _MODELS[model_name]


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _repair_json_like(text: str) -> dict[str, Any] | None:
    fence = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _generate_text(model_name: str, user_prompt: str) -> str:
    model, tokenizer = _get_model_and_tokenizer(model_name)
    # Chat template when available (Qwen2.5 Instruct, Phi, etc.)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_prompt

    inputs = tokenizer(text, return_tensors="pt")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only new tokens
    gen_only = out_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_only, skip_special_tokens=True)


def generate_llm_plans(
    *,
    model_name: str,
    goal: str,
    round2: list[dict[str, Any]],
    evidence_ids: list[str],
    evidence_snippets: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    snippets = evidence_snippets or []
    snippets_txt = json.dumps(snippets[:10], ensure_ascii=False)[:6000]
    prompt = (
        "You are an e-commerce decision orchestrator.\n"
        "Output STRICT JSON only. Do NOT use the word 'string' as a value.\n"
        "IMPORTANT: Each plan's actions, expected_impact, and risk MUST reflect the BUSINESS GOAL below "
        "and the retrieved products — not generic advice. Use different wording when the goal changes.\n\n"
        f"BUSINESS GOAL:\n{goal}\n\n"
        "Retrieved products (only use these product_id values for impacted_skus and evidence_refs):\n"
        f"{snippets_txt}\n\n"
        "Allowed product_id list:\n"
        f"{json.dumps(evidence_ids[:24])}\n\n"
        "Agent summaries (JSON, truncated):\n"
        f"{json.dumps(round2)[:3200]}\n\n"
        "Return a JSON object with key 'plans' containing exactly 3 objects. Each plan must have:\n"
        "plan_id (plan_a, plan_b, plan_c), rank (1-3), actions (array of 1-3 short strings), "
        "impacted_skus, expected_impact, risk, confidence (0.55-0.92), evidence_refs, price_change_pct (number).\n"
        "Do not copy example phrases from anywhere — write fresh text tied to the goal.\n"
        "Return JSON only, no markdown, no explanation."
    )
    try:
        out = _generate_text(model_name, prompt)
        obj = _extract_json_obj(out) or _repair_json_like(out)
        if not obj:
            retry = (
                prompt
                + "\nYour previous output was not valid JSON. Output one minified JSON object only."
            )
            out2 = _generate_text(model_name, retry)
            obj = _extract_json_obj(out2) or _repair_json_like(out2)
        if not obj or "plans" not in obj or not isinstance(obj["plans"], list):
            return None, {"enabled": True, "model_name": model_name, "status": "invalid_json"}
        return obj["plans"], {"enabled": True, "model_name": model_name, "status": "ok"}
    except Exception as exc:
        return None, {"enabled": True, "model_name": model_name, "status": f"error:{type(exc).__name__}"}
