from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from runtime.orchestrator import (
    OrchestratorConfig,
    OrchestratorContext,
    _load_inventory,
    _load_product_signals,
    _load_products,
    _load_retriever_from_index_meta,
    _load_sales_summary,
    _read_registry,
    build_ranked_plans,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--owner-id", required=True)
    ap.add_argument("--goals-json", type=Path, required=True, help="JSON list of goal strings.")
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--registry-path", type=Path, default=Path("copilot-v2/artifacts/registry/registry.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("copilot-v2/artifacts/evals"))
    ap.add_argument("--top-n-actions", type=int, default=3)
    ap.add_argument("--candidate-pool", type=int, default=40)
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--enable-pricing", action="store_true")
    ap.add_argument("--enable-sentiment", action="store_true")
    ap.add_argument("--write-docs-sample", action="store_true", help="Also write a tiny sample JSON under copilot-v2/docs/")
    ap.add_argument("--docs-sample-path", type=Path, default=Path("copilot-v2/docs/debate_replay_sample.json"))
    ap.add_argument("--docs-sample-n", type=int, default=3)
    args = ap.parse_args()

    goals = json.loads(Path(args.goals_json).read_text(encoding="utf-8"))
    if not isinstance(goals, list) or not all(isinstance(x, str) for x in goals):
        raise ValueError("--goals-json must be a JSON list of strings")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / str(args.snapshot_id) / "debate_replay" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "specialist_inputs.jsonl"

    registry = _read_registry(Path(args.registry_path))
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

    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        for goal in goals:
            g = str(goal).strip()
            if not g:
                continue
            out = build_ranked_plans(
                cfg=cfg,
                goal=g,
                owner_id=str(args.owner_id),
                horizon_days=int(args.horizon_days),
                enable_pricing=bool(args.enable_pricing),
                enable_sentiment=bool(args.enable_sentiment),
                ctx=ctx,
            )
            baseline_actions = out.get("ranked_actions") or []
            row = {
                "schema_version": 1,
                "snapshot_id": str(args.snapshot_id),
                "owner_id": str(args.owner_id),
                "goal": g,
                "constraints": out.get("input", {}).get("constraints") if isinstance(out.get("input"), dict) else {},
                "candidates": baseline_actions,
                "baseline_actions": baseline_actions[: int(args.top_n_actions)],
                "trace": out.get("trace") if isinstance(out.get("trace"), dict) else {},
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)

    if bool(args.write_docs_sample):
        sample = rows[: int(args.docs_sample_n)]
        Path(args.docs_sample_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.docs_sample_path).write_text(json.dumps(sample, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_path": str(out_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()

