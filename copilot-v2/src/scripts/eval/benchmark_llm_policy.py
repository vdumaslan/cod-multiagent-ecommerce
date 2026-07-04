from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from llm.ollama_client import OllamaClient
from runtime.debate import DebateConfig, run_chain_of_debate_safe
from runtime.orchestrator import (
    OrchestratorConfig,
    OrchestratorContext,
    _load_inventory,
    _load_products,
    _load_product_signals,
    _load_retriever_from_index_meta,
    _load_sales_summary,
    _read_registry,
    build_ranked_plans,
)


@dataclass(frozen=True)
class RunResult:
    ok: bool
    model: str
    goal: str
    owner_id: str
    baseline_n: int
    final_n: int
    debate_ok: bool
    debate_err: str | None
    latency_s: float
    judge_elapsed_s: float | None
    overlap_with_baseline: float | None


def _pids(actions: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for a in actions or []:
        pid = str((a or {}).get("product_id") or "").strip()
        if pid:
            out.append(pid)
    return out


def _overlap_at_k(a: list[str], b: list[str], k: int) -> float | None:
    if not a or not b or k <= 0:
        return None
    aa = set(a[:k])
    bb = set(b[:k])
    if not aa:
        return None
    return float(len(aa & bb)) / float(len(aa))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", type=str, required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--registry-path", type=Path, default=Path("copilot-v2/artifacts/registry/registry.json"))
    ap.add_argument("--owner-id", type=str, required=True)
    ap.add_argument("--models", type=str, default="qwen2.5:7b-instruct,llama3.1:8b,mistral:7b-instruct-v0.3-q4_K_M")
    ap.add_argument("--goals-json", type=Path, required=True, help="JSON list of goal strings (100+ recommended).")
    ap.add_argument("--out-dir", type=Path, default=Path("copilot-v2/artifacts/evals"))
    ap.add_argument("--top-n-actions", type=int, default=3)
    ap.add_argument("--candidate-pool", type=int, default=40)
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--candidate-m", type=int, default=20)
    ap.add_argument("--debate-top-k", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-goals", type=int, default=0, help="If >0, cap number of goals evaluated.")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N runs.")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    args = ap.parse_args()

    goals = json.loads(args.goals_json.read_text(encoding="utf-8"))
    if not isinstance(goals, list) or not all(isinstance(x, str) for x in goals):
        raise ValueError("--goals-json must be a JSON list of strings")
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / str(args.snapshot_id) / "llm_policy_benchmark" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = _read_registry(Path(args.registry_path))
    # owner-scoped index meta path
    index_meta = Path(args.artifacts_root) / "indexes" / str(args.snapshot_id) / "owners" / str(args.owner_id) / "dense" / "intfloat_e5-large-v2" / "index_meta.json"
    retriever = _load_retriever_from_index_meta(index_meta)

    ctx = OrchestratorContext(
        registry=registry,
        retriever=retriever,
        products=_load_products(artifacts_root=Path(args.artifacts_root), snapshot_id=str(args.snapshot_id)),
        signals=_load_product_signals(artifacts_root=Path(args.artifacts_root), snapshot_id=str(args.snapshot_id)),
        inventory=_load_inventory(artifacts_root=Path(args.artifacts_root), snapshot_id=str(args.snapshot_id)),
        sales_summary=_load_sales_summary(artifacts_root=Path(args.artifacts_root), snapshot_id=str(args.snapshot_id)),
    )

    cfg = OrchestratorConfig(
        snapshot_id=str(args.snapshot_id),
        artifacts_root=Path(args.artifacts_root),
        registry_path=Path(args.registry_path),
        top_n_actions=int(args.top_n_actions),
        candidate_pool=int(args.candidate_pool),
    )

    ollama = OllamaClient(base_url=str(args.ollama_url), timeout_s=60.0, retries=1)

    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"

    rows: list[RunResult] = []

    run_counter = 0
    goal_counter = 0
    for goal in goals:
        goal = str(goal).strip()
        if not goal:
            continue
        goal_counter += 1
        if int(args.max_goals) > 0 and goal_counter > int(args.max_goals):
            break
        for model in models:
            for rep in range(int(args.repeats)):
                run_counter += 1
                if int(args.progress_every) > 0 and (run_counter % int(args.progress_every) == 0):
                    print(json.dumps({"progress": True, "run": run_counter, "goal": goal, "model": model, "rep": rep}))
                t0 = time.time()
                baseline = build_ranked_plans(
                    cfg=cfg,
                    goal=goal,
                    owner_id=str(args.owner_id),
                    horizon_days=int(args.horizon_days),
                    enable_pricing=True,
                    enable_sentiment=True,
                    ctx=ctx,
                )
                baseline_actions = baseline.get("ranked_actions") or []
                baseline_pids = _pids(baseline_actions)

                final_actions = baseline_actions
                debate_ok = False
                debate_err = None
                judge_elapsed_s = None

                if baseline_actions:
                    candidates = baseline_actions[: max(3, int(args.candidate_m))]
                    acts, trace, err = run_chain_of_debate_safe(
                        ollama=ollama,
                        cfg=DebateConfig(model=str(model), top_k=int(args.debate_top_k), candidate_m=int(args.candidate_m)),
                        goal=goal,
                        constraints={},
                        candidates=candidates,
                        baseline_actions=baseline_actions[: int(args.debate_top_k)],
                    )
                    if acts is not None and err is None:
                        debate_ok = True
                        # merge like the server does (keep grounding)
                        by_pid = {str(a.get("product_id")): a for a in baseline_actions}
                        merged: list[dict[str, Any]] = []
                        for i, act in enumerate(acts):
                            pid = str(act["product_id"])
                            base = dict(by_pid.get(pid, {"product_id": pid}))
                            base["rank"] = i + 1
                            base["action_type"] = act["action_type"]
                            base["recommended_price_change_pct"] = float(act["recommended_price_change_pct"])
                            base["llm_rationale_bullets"] = act.get("rationale_bullets", [])
                            base["llm_risk_bullets"] = act.get("risk_bullets", [])
                            merged.append(base)
                        final_actions = merged
                        jr = (trace or {}).get("judge_raw") if isinstance(trace, dict) else None
                        if isinstance(jr, dict) and isinstance(jr.get("elapsed_s"), (int, float)):
                            judge_elapsed_s = float(jr["elapsed_s"])
                    else:
                        debate_ok = False
                        debate_err = str(err or "unknown")

                latency_s = float(time.time() - t0)
                final_pids = _pids(final_actions)
                overlap = _overlap_at_k(final_pids, baseline_pids, k=min(len(final_pids), len(baseline_pids), int(args.top_n_actions)))

                r = RunResult(
                    ok=True,
                    model=str(model),
                    goal=goal,
                    owner_id=str(args.owner_id),
                    baseline_n=len(baseline_actions),
                    final_n=len(final_actions),
                    debate_ok=debate_ok,
                    debate_err=debate_err,
                    latency_s=latency_s,
                    judge_elapsed_s=judge_elapsed_s,
                    overlap_with_baseline=overlap,
                )
                rows.append(r)

    # write rows.jsonl
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__) + "\n")

    # summarize
    by_model: dict[str, list[RunResult]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    def _p95(xs: list[float]) -> float | None:
        if not xs:
            return None
        xs2 = sorted(xs)
        i = int(round(0.95 * (len(xs2) - 1)))
        return float(xs2[i])

    summary: dict[str, Any] = {"snapshot_id": str(args.snapshot_id), "owner_id": str(args.owner_id), "n_rows": len(rows), "by_model": {}}
    for m, rs in sorted(by_model.items()):
        lat = [x.latency_s for x in rs]
        judge_lat = [x.judge_elapsed_s for x in rs if x.judge_elapsed_s is not None]
        succ = [1.0 if x.debate_ok else 0.0 for x in rs]
        ov = [x.overlap_with_baseline for x in rs if x.overlap_with_baseline is not None]
        summary["by_model"][m] = {
            "n": len(rs),
            "success_rate": float(mean(succ)) if succ else None,
            "avg_latency_s": float(mean(lat)) if lat else None,
            "p95_latency_s": _p95(lat),
            "avg_judge_elapsed_s": float(mean(judge_lat)) if judge_lat else None,
            "avg_overlap_with_baseline": float(mean(ov)) if ov else None,
        }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(str(summary_path))


if __name__ == "__main__":
    main()

