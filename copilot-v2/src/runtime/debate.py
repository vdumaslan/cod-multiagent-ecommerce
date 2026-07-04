from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing import Literal

from llm.json_schema import (
    SchemaError,
    extract_json_object,
    validate_peer_review,
    validate_ranked_actions,
    validate_specialist_proposal,
)
from llm.ollama_client import OllamaClient


def _extract_json_with_retries(
    *,
    ollama: OllamaClient,
    cfg: DebateConfig,
    messages: list[dict[str, str]],
    repair_messages: list[str],
    trace_key: str,
    raw_out: dict[str, Any],
) -> dict[str, Any]:
    """Try extract_json_object; on failure retry with repair prompts (reduces JSONDecodeError)."""
    res = ollama.chat(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_predict=cfg.num_predict,
        seed=cfg.seed,
    )
    raw_out[trace_key] = {"elapsed_s": res.elapsed_s, "content": res.content[:3000]}
    try:
        return extract_json_object(res.content)
    except Exception:
        pass
    for i, hint in enumerate(repair_messages):
        res2 = ollama.chat(
            model=cfg.model,
            messages=messages
            + [{"role": "user", "content": hint}],
            temperature=0.0,
            top_p=1.0,
            num_predict=cfg.num_predict,
            seed=cfg.seed + 20 + i,
        )
        raw_out[f"{trace_key}_retry_{i}"] = {"elapsed_s": res2.elapsed_s, "content": res2.content[:3000]}
        try:
            return extract_json_object(res2.content)
        except Exception:
            continue
    # Final attempt: short output only
    res3 = ollama.chat(
        model=cfg.model,
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "Output must be a single JSON object, under 800 characters, no prose, no markdown, no code fences. "
                    "Use double quotes for all keys and string values."
                ),
            }
        ],
        temperature=0.0,
        top_p=1.0,
        num_predict=min(512, cfg.num_predict),
        seed=cfg.seed + 99,
    )
    raw_out[f"{trace_key}_retry_final"] = {"elapsed_s": res3.elapsed_s, "content": res3.content[:3000]}
    return extract_json_object(res3.content)


def _fallback_specialist_proposal(
    *,
    cand: list[dict[str, Any]],
    allowed: set[str],
    top_k: int,
) -> dict[str, Any]:
    acts: list[dict[str, Any]] = []
    for c in cand[: int(top_k)]:
        pid = str((c or {}).get("product_id") or "").strip()
        if pid and pid in allowed:
            acts.append(
                {
                    "product_id": pid,
                    "action_type": "reprice",
                    "recommended_price_change_pct": float((c or {}).get("recommended_price_change_pct") or 0.0),
                }
            )
        if len(acts) >= int(top_k):
            break
    return {
        "proposed_actions": acts,
        "key_claims": ["Fallback: specialist JSON did not validate; using candidate summaries."],
        "concerns": [],
    }


def _fallback_ranked_from_baseline(
    *,
    baseline_actions: list[dict[str, Any]],
    allowed: set[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], bool]:
    """If judge JSON fails, use grounded baseline rows (same product_ids only)."""
    out: list[dict[str, Any]] = []
    for a in baseline_actions[: int(top_k)]:
        pid = str((a or {}).get("product_id") or "").strip()
        if not pid or pid not in allowed:
            continue
        out.append(
            {
                "product_id": pid,
                "action_type": str((a or {}).get("action_type") or "reprice").strip().lower() or "reprice",
                "recommended_price_change_pct": float((a or {}).get("recommended_price_change_pct") or 0.0),
                "rationale_bullets": ["Fallback: using grounded baseline plan (LLM judge output did not validate)."],
                "risk_bullets": ["Model JSON/schema validation failed; baseline actions preserved."],
            }
        )
        if len(out) >= int(top_k):
            break
    if not out:
        raise SchemaError("No valid ranked_actions after validation.")
    return out, True


@dataclass(frozen=True)
class DebateConfig:
    model: str
    top_k: int = 3
    candidate_m: int = 20
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 700
    seed: int = 42


def _system(role: str) -> str:
    return (
        "You are an orchestrator specialist role. "
        "You must be concise, grounded in provided candidate facts only, and output STRICT JSON only."
        f" Role={role}."
    )


def _proposal_prompt(role: str, *, payload: dict[str, Any], top_k: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system(role)},
        {
            "role": "user",
            "content": (
                "Task: propose top actions.\n"
                f"Return JSON with keys: proposed_actions (top {top_k}), key_claims (<=5), concerns (<=5).\n"
                "Each proposed action must include product_id, action_type, recommended_price_change_pct.\n"
                "Use only product_ids from candidates.\n\n"
                "Output JSON only. Do not wrap in markdown. Do not include any extra text.\n"
                f"INPUT_JSON:\n{payload}"
            ),
        },
    ]


def _peer_review_prompt(role: str, *, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system(role)},
        {
            "role": "user",
            "content": (
                "Task: peer-review other specialists and suggest changes.\n"
                "Return JSON with keys: agreements (<=6), disagreements (<=6), suggested_changes (<=6).\n\n"
                "Output JSON only. Do not wrap in markdown. Do not include any extra text.\n"
                f"INPUT_JSON:\n{payload}"
            ),
        },
    ]


def _judge_prompt(*, payload: dict[str, Any], top_k: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Judge. You must synthesize specialists + peer reviews into a final decision.\n"
                "Output STRICT JSON only with key ranked_actions.\n"
                f"ranked_actions must be a list of top {top_k} items.\n"
                "Each item: product_id, action_type, recommended_price_change_pct, rationale_bullets (2-4), risk_bullets (1-3).\n"
                "Use only candidate product_ids. Respect constraints: max_abs_price_change_pct.\n"
                "Output JSON only. Do not wrap in markdown. Do not include any extra text.\n"
            ),
        },
        {"role": "user", "content": f"INPUT_JSON:\n{payload}"},
    ]


def run_chain_of_debate(
    *,
    ollama: OllamaClient,
    cfg: DebateConfig,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cand = candidates[: int(cfg.candidate_m)]
    allowed = {str(c.get("product_id")) for c in cand if c.get("product_id")}

    base_payload = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
    }

    roles = ["RetrievalSpecialist", "PricingSpecialist", "SentimentSpecialist"]
    proposals: dict[str, Any] = {}
    proposal_raw: dict[str, Any] = {}
    for role in roles:
        msgs = _proposal_prompt(role, payload=base_payload, top_k=cfg.top_k)
        try:
            obj = _extract_json_with_retries(
                ollama=ollama,
                cfg=cfg,
                messages=msgs,
                repair_messages=[
                    "Your previous output was invalid. Return ONE valid JSON object only.",
                    'Return JSON with keys: proposed_actions (array), key_claims (array), concerns (array). Example: {"proposed_actions":[],"key_claims":[],"concerns":[]}',
                ],
                trace_key=role,
                raw_out=proposal_raw,
            )
            proposals[role] = validate_specialist_proposal(obj, allowed_product_ids=allowed, top_k=cfg.top_k)
        except Exception:
            proposals[role] = _fallback_specialist_proposal(cand=cand, allowed=allowed, top_k=cfg.top_k)
            proposal_raw[role + "_fallback"] = True

    peer_reviews: dict[str, Any] = {}
    peer_raw: dict[str, Any] = {}
    for role in roles:
        others = {r: proposals[r] for r in roles if r != role}
        payload = {
            "goal": goal,
            "constraints": constraints,
            "candidates": cand,
            "your_role": role,
            "other_proposals": others,
        }
        msgs = _peer_review_prompt(role, payload=payload)
        try:
            obj = _extract_json_with_retries(
                ollama=ollama,
                cfg=cfg,
                messages=msgs,
                repair_messages=[
                    "Return ONE valid JSON object only.",
                    'Return JSON with keys: agreements, disagreements, suggested_changes (arrays of strings). Example: {"agreements":[],"disagreements":[],"suggested_changes":[]}',
                ],
                trace_key=f"peer_{role}",
                raw_out=peer_raw,
            )
            peer_reviews[role] = validate_peer_review(obj)
        except Exception:
            peer_reviews[role] = validate_peer_review({})
            peer_raw[f"peer_{role}_fallback"] = True

    judge_payload = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
        "specialist_proposals": proposals,
        "peer_reviews": peer_reviews,
    }
    judge_messages = _judge_prompt(payload=judge_payload, top_k=cfg.top_k)
    allowed_hint = ", ".join(sorted(allowed)[:24])
    if len(allowed) > 24:
        allowed_hint += ", ..."

    judge_raw: dict[str, Any] = {}
    final_actions: list[dict[str, Any]] | None = None
    judge_fallback = False

    try:
        judge_obj = _extract_json_with_retries(
            ollama=ollama,
            cfg=cfg,
            messages=judge_messages,
            repair_messages=[
                "Return ONE JSON object only. Top-level key must be ranked_actions (array). No markdown.",
                (
                    'Example shape: {"ranked_actions":[{"product_id":"<ID>","action_type":"reprice",'
                    '"recommended_price_change_pct":0.0,"rationale_bullets":["a"],"risk_bullets":["b"]}]}'
                    f". Use only these product_id values: {allowed_hint}"
                ),
            ],
            trace_key="judge",
            raw_out=judge_raw,
        )
        final_actions = validate_ranked_actions(judge_obj, allowed_product_ids=allowed, top_k=cfg.top_k)
    except Exception as e1:
        judge_raw["validate_or_parse_error"] = str(e1)[:300]
        try:
            repair_user = (
                "Your output must validate as JSON with key ranked_actions. "
                f"Each item needs product_id (one of: {allowed_hint}), action_type, recommended_price_change_pct, "
                "rationale_bullets (2-4 strings), risk_bullets (1-3 strings). "
                f"Return at most {int(cfg.top_k)} items. No markdown, no commentary."
            )
            res_fix = ollama.chat(
                model=cfg.model,
                messages=judge_messages + [{"role": "user", "content": repair_user}],
                temperature=0.0,
                top_p=1.0,
                num_predict=cfg.num_predict,
                seed=cfg.seed + 31,
            )
            judge_raw["schema_repair"] = {"elapsed_s": res_fix.elapsed_s, "content": res_fix.content[:4000]}
            judge_obj2 = extract_json_object(res_fix.content)
            final_actions = validate_ranked_actions(judge_obj2, allowed_product_ids=allowed, top_k=cfg.top_k)
        except Exception as e2:
            judge_raw["schema_repair_error"] = str(e2)[:300]
            final_actions = None

    if final_actions is None:
        final_actions, judge_fallback = _fallback_ranked_from_baseline(
            baseline_actions=baseline_actions, allowed=allowed, top_k=cfg.top_k
        )
        judge_raw["used_baseline_fallback"] = True

    trace = {
        "model": cfg.model,
        "top_k": int(cfg.top_k),
        "candidate_m": int(cfg.candidate_m),
        "proposal_raw": proposal_raw,
        "peer_raw": peer_raw,
        "judge_raw": judge_raw,
        "judge_fallback": judge_fallback,
        "proposals": proposals,
        "peer_reviews": peer_reviews,
    }
    return final_actions, trace


def run_chain_of_debate_safe(**kwargs):
    try:
        acts, trace = run_chain_of_debate(**kwargs)
        return acts, trace, None
    except SchemaError as e:
        return None, {"error": "SchemaError", "message": str(e)}, "SchemaError"
    except Exception as e:  # noqa: BLE001
        return None, {"error": type(e).__name__, "message": str(e)[:500]}, type(e).__name__


@dataclass(frozen=True)
class DebateACJConfig:
    advocate_model: str
    critic_model: str
    judge_model: str
    prompt_style: Literal["zero_shot_json", "few_shot_json", "cot_hidden", "structured_rationale"] = "zero_shot_json"
    prompt_version: str = "v1"
    top_k: int = 3
    candidate_m: int = 20
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 800
    seed: int = 42


def _few_shot_block_for_schema(*, role: str, top_k: int) -> str:
    """
    Few-shot examples are schema-only (no real product_ids); they must remain identical across models.
    The model should treat them as formatting guidance, not as candidate content.
    """
    if role == "Advocate":
        return (
            "FEW_SHOT_EXAMPLE_1 (format only):\n"
            '{"proposed_actions":[{"product_id":"EXAMPLE_PID_1","action_type":"reprice","recommended_price_change_pct":2.5}],'
            '"key_claims":["..."],"concerns":["..."]}\n'
            "FEW_SHOT_EXAMPLE_2 (format only):\n"
            '{"proposed_actions":[{"product_id":"EXAMPLE_PID_2","action_type":"investigate","recommended_price_change_pct":0.0}],'
            '"key_claims":["..."],"concerns":["..."]}\n'
        )
    if role == "Critic":
        return (
            "FEW_SHOT_EXAMPLE_1 (format only):\n"
            '{"agreements":["..."],"disagreements":["..."],"suggested_changes":["..."]}\n'
        )
    # Judge
    return (
        "FEW_SHOT_EXAMPLE_1 (format only):\n"
        f'{{"ranked_actions":[{{"product_id":"EXAMPLE_PID_1","action_type":"reprice","recommended_price_change_pct":0.0,'
        '"rationale_bullets":["...","..."],"risk_bullets":["..."]}}]}}\n'
    )


def _advocate_prompt(*, payload: dict[str, Any], top_k: int, prompt_style: str, prompt_version: str) -> list[dict[str, str]]:
    style = str(prompt_style or "zero_shot_json").strip()
    version = str(prompt_version or "v1").strip()
    think = ""
    if style == "cot_hidden":
        think = (
            "Before answering, think step-by-step privately: identify constraints, conflicts, and the top grounded actions.\n"
            "Do NOT output your reasoning.\n"
        )
    few_shot = ""
    if style == "few_shot_json":
        few_shot = (
            "Below are FORMAT-ONLY examples. They are NOT candidate product_ids.\n"
            + _few_shot_block_for_schema(role="Advocate", top_k=top_k)
            + "\n"
        )
    return [
        {"role": "system", "content": _system("Advocate")},
        {
            "role": "user",
            "content": "".join(
                [
                    f"prompt_style={style} prompt_version={version}\n",
                    "Task: argue for the strongest grounded action plan.\n",
                    think,
                    few_shot,
                    "Return STRICT JSON only.\n",
                    f"Return JSON with keys: proposed_actions (top {top_k}), key_claims (<=5), concerns (<=5).\n",
                    "Each proposed action must include product_id, action_type, recommended_price_change_pct.\n",
                    "Use only product_ids from candidates.\n\n",
                    "Output JSON only. Do not wrap in markdown. Do not include any extra text.\n",
                    f"INPUT_JSON:\n{payload}",
                ]
            ),
        },
    ]


def _critic_prompt(*, payload: dict[str, Any], prompt_style: str, prompt_version: str) -> list[dict[str, str]]:
    style = str(prompt_style or "zero_shot_json").strip()
    version = str(prompt_version or "v1").strip()
    think = ""
    if style == "cot_hidden":
        think = (
            "Before answering, think step-by-step privately: find constraint violations, ungrounded IDs, and specialist conflicts.\n"
            "Do NOT output your reasoning.\n"
        )
    few_shot = ""
    if style == "few_shot_json":
        few_shot = (
            "Below are FORMAT-ONLY examples.\n"
            + _few_shot_block_for_schema(role="Critic", top_k=0)
            + "\n"
        )
    return [
        {"role": "system", "content": _system("Critic")},
        {
            "role": "user",
            "content": "".join(
                [
                    f"prompt_style={style} prompt_version={version}\n",
                    "Task: challenge the Advocate plan using the same grounded inputs.\n",
                    "Return JSON with keys: agreements (<=6), disagreements (<=6), suggested_changes (<=6).\n",
                    "Focus on risks, missing constraints, and conflicts between specialists (pricing/sentiment/inventory/retrieval).\n\n",
                    think,
                    few_shot,
                    "Output JSON only. Do not wrap in markdown. Do not include any extra text.\n",
                    f"INPUT_JSON:\n{payload}",
                ]
            ),
        },
    ]


def _judge_prompt_acj(*, payload: dict[str, Any], top_k: int, prompt_style: str, prompt_version: str) -> list[dict[str, str]]:
    # We still reuse the same schema validator; this only changes *instructions*.
    style = str(prompt_style or "zero_shot_json").strip()
    version = str(prompt_version or "v1").strip()
    think = ""
    if style == "cot_hidden":
        think = (
            "Before answering, think step-by-step privately: reconcile advocate vs critic, enforce constraints, select grounded IDs.\n"
            "Do NOT output your reasoning.\n"
        )
    few_shot = ""
    if style == "few_shot_json":
        few_shot = (
            "Below are FORMAT-ONLY examples. They are NOT candidate product_ids.\n"
            + _few_shot_block_for_schema(role="Judge", top_k=top_k)
            + "\n"
        )
    if style == "structured_rationale":
        # Stronger emphasis on concise, structured bullets (safer than free-form CoT).
        extra = (
            "Write rationale_bullets as short, grounded evidence statements (2â€“4 bullets).\n"
            "Write risk_bullets as constraint/risk checks (1â€“3 bullets).\n"
        )
    else:
        extra = ""

    msgs = _judge_prompt(payload=payload, top_k=top_k)
    # Insert a lightweight user preface right before the payload.
    preface = (
        f"prompt_style={style} prompt_version={version}\n"
        + think
        + few_shot
        + extra
        + "Return JSON only.\n"
    )
    # msgs is: [system, user(payload)]. We add preface as another user message before payload.
    return [msgs[0], {"role": "user", "content": preface}, msgs[1]]


def run_advocate_critic_judge(
    *,
    ollama: OllamaClient,
    cfg: DebateACJConfig,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Debate v2: Advocate -> Critic -> Judge.

    Uses the same strict JSON extraction/repair logic and the existing validators:
    - Advocate: validate_specialist_proposal
    - Critic: validate_peer_review
    - Judge: validate_ranked_actions
    """
    cand = candidates[: int(cfg.candidate_m)]
    allowed = {str(c.get("product_id")) for c in cand if c.get("product_id")}

    base_payload = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
    }

    raw: dict[str, Any] = {}

    # --- Advocate ---
    advocate_msgs = _advocate_prompt(
        payload=base_payload,
        top_k=cfg.top_k,
        prompt_style=cfg.prompt_style,
        prompt_version=cfg.prompt_version,
    )
    advocate_obj = _extract_json_with_retries(
        ollama=ollama,
        cfg=DebateConfig(
            model=cfg.advocate_model,
            top_k=cfg.top_k,
            candidate_m=cfg.candidate_m,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_predict=cfg.num_predict,
            seed=cfg.seed,
        ),
        messages=advocate_msgs,
        repair_messages=[
            "Return ONE valid JSON object only.",
            'Return JSON with keys: proposed_actions, key_claims, concerns. Example: {"proposed_actions":[],"key_claims":[],"concerns":[]}',
        ],
        trace_key="advocate",
        raw_out=raw,
    )
    try:
        advocate = validate_specialist_proposal(advocate_obj, allowed_product_ids=allowed, top_k=cfg.top_k)
    except Exception:
        advocate = _fallback_specialist_proposal(cand=cand, allowed=allowed, top_k=cfg.top_k)
        raw["advocate_fallback"] = True

    # --- Critic ---
    critic_payload = {**base_payload, "advocate": advocate}
    critic_msgs = _critic_prompt(payload=critic_payload, prompt_style=cfg.prompt_style, prompt_version=cfg.prompt_version)
    critic_obj = _extract_json_with_retries(
        ollama=ollama,
        cfg=DebateConfig(
            model=cfg.critic_model,
            top_k=cfg.top_k,
            candidate_m=cfg.candidate_m,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_predict=cfg.num_predict,
            seed=cfg.seed + 7,
        ),
        messages=critic_msgs,
        repair_messages=[
            "Return ONE valid JSON object only.",
            'Return JSON with keys: agreements, disagreements, suggested_changes. Example: {"agreements":[],"disagreements":[],"suggested_changes":[]}',
        ],
        trace_key="critic",
        raw_out=raw,
    )
    try:
        critic = validate_peer_review(critic_obj)
    except Exception:
        critic = validate_peer_review({})
        raw["critic_fallback"] = True

    # --- Judge ---
    judge_payload = {**base_payload, "advocate": advocate, "critic": critic, "debate_history": {"advocate": advocate, "critic": critic}}
    judge_msgs = _judge_prompt_acj(
        payload=judge_payload,
        top_k=cfg.top_k,
        prompt_style=cfg.prompt_style,
        prompt_version=cfg.prompt_version,
    )

    allowed_hint = ", ".join(sorted(allowed)[:24])
    if len(allowed) > 24:
        allowed_hint += ", ..."

    judge_raw: dict[str, Any] = {}
    final_actions: list[dict[str, Any]] | None = None
    judge_fallback = False
    try:
        judge_obj = _extract_json_with_retries(
            ollama=ollama,
            cfg=DebateConfig(
                model=cfg.judge_model,
                top_k=cfg.top_k,
                candidate_m=cfg.candidate_m,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                num_predict=cfg.num_predict,
                seed=cfg.seed + 21,
            ),
            messages=judge_msgs,
            repair_messages=[
                "Return ONE JSON object only. Top-level key must be ranked_actions (array). No markdown.",
                (
                    'Example shape: {"ranked_actions":[{"product_id":"<ID>","action_type":"reprice",'
                    '"recommended_price_change_pct":0.0,"rationale_bullets":["a"],"risk_bullets":["b"]}]}'
                    f". Use only these product_id values: {allowed_hint}"
                ),
            ],
            trace_key="judge",
            raw_out=judge_raw,
        )
        final_actions = validate_ranked_actions(judge_obj, allowed_product_ids=allowed, top_k=cfg.top_k)
    except Exception as e1:
        judge_raw["validate_or_parse_error"] = str(e1)[:300]
        final_actions = None

    if final_actions is None:
        final_actions, judge_fallback = _fallback_ranked_from_baseline(
            baseline_actions=baseline_actions, allowed=allowed, top_k=cfg.top_k
        )
        judge_raw["used_baseline_fallback"] = True

    trace = {
        "mode": "acj",
        "advocate_model": cfg.advocate_model,
        "critic_model": cfg.critic_model,
        "judge_model": cfg.judge_model,
        "prompt_style": cfg.prompt_style,
        "prompt_version": cfg.prompt_version,
        "top_k": int(cfg.top_k),
        "candidate_m": int(cfg.candidate_m),
        "raw": raw,
        "judge_raw": judge_raw,
        "judge_fallback": bool(judge_fallback),
        "advocate": advocate,
        "critic": critic,
    }
    return final_actions, trace


def run_advocate_critic_judge_safe(**kwargs):
    try:
        acts, trace = run_advocate_critic_judge(**kwargs)
        return acts, trace, None
    except SchemaError as e:
        return None, {"error": "SchemaError", "message": str(e)}, "SchemaError"
    except Exception as e:  # noqa: BLE001
        return None, {"error": type(e).__name__, "message": str(e)[:500]}, type(e).__name__

