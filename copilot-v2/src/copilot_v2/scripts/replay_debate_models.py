from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from copilot_v2.eval.debate_rubric import score_debate_output
from copilot_v2.llm.ollama_client import OllamaClient
from copilot_v2.runtime.debate import DebateACJConfig, run_advocate_critic_judge_safe


@dataclass(frozen=True)
class ReplayRow:
    ok: bool
    advocate_model: str
    critic_model: str
    judge_model: str
    prompt_style: str
    prompt_version: str
    goal: str
    owner_id: str
    snapshot_id: str
    latency_s: float
    debate_ok: bool
    debate_err: str | None
    rubric: dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-jsonl", type=Path, required=True, help="specialist_inputs.jsonl from cache_specialist_outputs.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--advocate-models", type=str, required=True, help="Comma-separated model list")
    ap.add_argument("--critic-models", type=str, required=True, help="Comma-separated model list")
    ap.add_argument("--judge-models", type=str, required=True, help="Comma-separated model list")
    ap.add_argument("--candidate-m", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--prompt-style",
        type=str,
        default="zero_shot_json",
        help="Prompt variant for ACJ debate: zero_shot_json|few_shot_json|cot_hidden|structured_rationale",
    )
    ap.add_argument("--prompt-version", type=str, default="v1")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    advocate_models = [m.strip() for m in args.advocate_models.split(",") if m.strip()]
    critic_models = [m.strip() for m in args.critic_models.split(",") if m.strip()]
    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"

    # Load inputs
    inputs: list[dict[str, Any]] = []
    with Path(args.inputs_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inputs.append(json.loads(line))

    ollama = OllamaClient(base_url=str(args.ollama_url), timeout_s=90.0, retries=1)

    rows: list[ReplayRow] = []
    run_counter = 0
    # Write incrementally so long runs are observable and crash-safe.
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = rows_path.open("w", encoding="utf-8")
    for inp in inputs:
        goal = str(inp.get("goal") or "")
        owner_id = str(inp.get("owner_id") or "")
        snapshot_id = str(inp.get("snapshot_id") or "")
        constraints = inp.get("constraints") if isinstance(inp.get("constraints"), dict) else {}
        candidates = inp.get("candidates") if isinstance(inp.get("candidates"), list) else []
        baseline_actions = inp.get("baseline_actions") if isinstance(inp.get("baseline_actions"), list) else []

        allowed = {str(c.get("product_id")) for c in candidates if isinstance(c, dict) and c.get("product_id")}
        max_abs = float((constraints or {}).get("max_abs_price_change_pct") or 10.0)

        for am in advocate_models:
            for cm in critic_models:
                for jm in judge_models:
                    run_counter += 1
                    if int(args.progress_every) > 0 and run_counter % int(args.progress_every) == 0:
                        print(
                            json.dumps(
                                {
                                    "progress": True,
                                    "run": run_counter,
                                    "goal": goal,
                                    "advocate": am,
                                    "critic": cm,
                                    "judge": jm,
                                    "prompt_style": str(args.prompt_style),
                                    "prompt_version": str(args.prompt_version),
                                }
                            ),
                            flush=True,
                        )

                    t0 = time.time()
                    acts, trace, err = run_advocate_critic_judge_safe(
                        ollama=ollama,
                        cfg=DebateACJConfig(
                            advocate_model=am,
                            critic_model=cm,
                            judge_model=jm,
                            prompt_style=str(args.prompt_style),
                            prompt_version=str(args.prompt_version),
                            top_k=int(args.top_k),
                            candidate_m=int(args.candidate_m),
                        ),
                        goal=goal,
                        constraints=constraints or {},
                        candidates=candidates,
                        baseline_actions=baseline_actions,
                    )
                    latency_s = float(time.time() - t0)

                    critic = None
                    if isinstance(trace, dict):
                        critic = trace.get("critic") if isinstance(trace.get("critic"), dict) else None

                    rubric = score_debate_output(
                        final_actions=acts,
                        allowed_product_ids=allowed,
                        max_abs_price_change_pct=max_abs,
                        candidates_by_pid={str(c.get("product_id")): c for c in candidates if isinstance(c, dict) and c.get("product_id")},
                        critic=critic,
                    )

                    rows.append(
                        ReplayRow(
                            ok=True,
                            advocate_model=am,
                            critic_model=cm,
                            judge_model=jm,
                            prompt_style=str(args.prompt_style),
                            prompt_version=str(args.prompt_version),
                            goal=goal,
                            owner_id=owner_id,
                            snapshot_id=snapshot_id,
                            latency_s=latency_s,
                            debate_ok=acts is not None and err is None,
                            debate_err=str(err) if err else None,
                            rubric=rubric,
                        )
                    )
                    out_f.write(json.dumps(rows[-1].__dict__) + "\n")
                    out_f.flush()

    out_f.close()

    # Summarize by combo
    by_combo: dict[str, list[ReplayRow]] = {}
    for r in rows:
        k = f"{r.advocate_model} | {r.critic_model} | {r.judge_model} | {r.prompt_style}:{r.prompt_version}"
        by_combo.setdefault(k, []).append(r)

    def _avg(key: str, rs: list[ReplayRow]) -> float | None:
        xs: list[float] = []
        for r in rs:
            v = r.rubric.get(key)
            if isinstance(v, (int, float)):
                xs.append(float(v))
        return float(mean(xs)) if xs else None

    summary: dict[str, Any] = {"n_rows": len(rows), "by_combo": {}}
    for k, rs in sorted(by_combo.items()):
        summary["by_combo"][k] = {
            "n": len(rs),
            "debate_ok_rate": float(sum(1 for x in rs if x.debate_ok)) / float(len(rs) or 1),
            "latency_s_mean": float(mean([x.latency_s for x in rs])) if rs else None,
            "rubric_mean": {
                "json_valid": _avg("json_valid", rs),
                "grounded_ids": _avg("grounded_ids", rs),
                "constraint_respect": _avg("constraint_respect", rs),
                "conflict_detection": _avg("conflict_detection", rs),
                "decision_justification": _avg("decision_justification", rs),
            },
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "rows_path": str(rows_path), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

