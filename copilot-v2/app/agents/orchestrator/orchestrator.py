"""Advocate → Critic rounds (N times) then Judge once on Move On."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.llm import OllamaClient
from app.agents.orchestrator import advocate, critic, judge


@dataclass(frozen=True)
class ACJConfig:
    advocate_model: str = "llama3.1:8b"
    critic_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:7b-instruct"
    prompt_style: str = "few_shot_json"
    prompt_version: str = "v1"
    top_k: int = 3
    candidate_m: int = 20
    temperature: float = 0.2
    num_predict: int = 800
    seed: int = 42


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _run_round(
    ollama: OllamaClient,
    *,
    cfg: ACJConfig,
    base_payload: dict[str, Any],
    seed_offset: int = 0,
    is_revision: bool = False,
    run_dir: Path | None = None,
    adv_filename: str = "",
    crit_filename: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one Advocate + Critic round. Returns (adv_result, crit_result, adv_raw, crit_raw)."""
    adv_result, adv_raw = advocate.run(
        ollama,
        model=cfg.advocate_model,
        payload=base_payload,
        top_k=cfg.top_k,
        prompt_style=cfg.prompt_style,
        prompt_version=cfg.prompt_version,
        seed=cfg.seed + seed_offset,
        temperature=cfg.temperature,
        num_predict=cfg.num_predict,
        is_revision=is_revision,
    )
    if run_dir and adv_filename:
        _write(run_dir / adv_filename, {"role": "advocate", "is_revision": is_revision, "result": adv_result, "raw": adv_raw})

    crit_payload = {**base_payload, "advocate": adv_result}
    crit_result, crit_raw = critic.run(
        ollama,
        model=cfg.critic_model,
        payload=crit_payload,
        prompt_style=cfg.prompt_style,
        prompt_version=cfg.prompt_version,
        seed=cfg.seed + 7 + seed_offset,
        temperature=cfg.temperature,
        num_predict=cfg.num_predict,
    )
    if run_dir and crit_filename:
        _write(run_dir / crit_filename, {"role": "critic", "is_revision": is_revision, "result": crit_result, "raw": crit_raw})

    return adv_result, crit_result, adv_raw, crit_raw


def run_acj(
    ollama: OllamaClient,
    *,
    cfg: ACJConfig,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_actions: list[dict[str, Any]],
    run_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Round 1: Advocate → Critic only. Returns (adv_result, crit_result, debate_trace)."""
    cand = candidates[: cfg.candidate_m]
    base_payload: dict[str, Any] = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
    }

    adv_result, crit_result, adv_raw, crit_raw = _run_round(
        ollama,
        cfg=cfg,
        base_payload=base_payload,
        seed_offset=0,
        is_revision=False,
        run_dir=run_dir,
        adv_filename="4_debate_advocate_r1.json",
        crit_filename="5_debate_critic_r1.json",
    )

    return adv_result, crit_result, {
        "mode": "acj",
        "advocate_model": cfg.advocate_model,
        "critic_model": cfg.critic_model,
        "prompt_style": cfg.prompt_style,
        "prompt_version": cfg.prompt_version,
        "top_k": cfg.top_k,
        "candidate_m": cfg.candidate_m,
        "raw": {**adv_raw, **crit_raw},
        "advocate": adv_result,
        "critic": crit_result,
    }


def continue_acj(
    ollama: OllamaClient,
    *,
    cfg: ACJConfig,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_actions: list[dict[str, Any]],
    prev_advocate: dict[str, Any],
    prev_critic: dict[str, Any],
    human_feedback: str | None = None,
    run_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One additional Advocate + Critic round (no judge). Returns (adv_result, crit_result, raw_trace)."""
    cand = candidates[: cfg.candidate_m]
    payload: dict[str, Any] = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
        "round1_advocate": prev_advocate,
        "round1_critic": prev_critic,
    }
    if human_feedback:
        payload["human_feedback"] = human_feedback

    adv_result, crit_result, adv_raw, crit_raw = _run_round(
        ollama,
        cfg=cfg,
        base_payload=payload,
        seed_offset=100,
        is_revision=True,
        run_dir=run_dir,
        adv_filename="4_debate_advocate.json",
        crit_filename="5_debate_critic.json",
    )

    return adv_result, crit_result, {**adv_raw, **crit_raw}


def run_judge_only(
    ollama: OllamaClient,
    *,
    cfg: ACJConfig,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_actions: list[dict[str, Any]],
    latest_advocate: dict[str, Any],
    latest_critic: dict[str, Any],
    human_feedback: str | None = None,
    run_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Run the judge once using the final advocate + critic. Returns (final_actions, judge_raw, fallback)."""
    cand = candidates[: cfg.candidate_m]
    judge_payload: dict[str, Any] = {
        "goal": goal,
        "constraints": constraints,
        "candidates": cand,
        "baseline_actions": baseline_actions,
        "advocate": latest_advocate,
        "critic": latest_critic,
    }
    if human_feedback:
        judge_payload["human_feedback"] = human_feedback

    final_actions, judge_raw, judge_fallback = judge.run(
        ollama,
        model=cfg.judge_model,
        payload=judge_payload,
        baseline_actions=baseline_actions,
        top_k=cfg.top_k,
        prompt_style=cfg.prompt_style,
        prompt_version=cfg.prompt_version,
        seed=cfg.seed + 21,
        temperature=cfg.temperature,
        num_predict=cfg.num_predict,
    )
    if run_dir:
        _write(run_dir / "8_debate_judge.json", {
            "role": "judge",
            "result": final_actions,
            "raw": judge_raw,
            "fallback": judge_fallback,
        })

    return final_actions, judge_raw, judge_fallback
