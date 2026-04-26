from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Recreated entrypoint for the FT-Transformer pricing lane.

    The original run produced `ft_transformer_tune_report.json` and `ft_transformer_train_report.json`
    under `copilot-v2/artifacts/evals/<snapshot_id>/pricing/`.

    This implementation is intentionally conservative: it writes a tune-report skeleton and fails
    fast with a clear message if the required deep-learning training code is not available.
    """

    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--device", default="auto")
    p.add_argument("--policy-bound", type=float, default=10.0)
    p.add_argument("--val-screen-rows", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--grid-screen-epochs", type=int, default=28)
    p.add_argument("--grid-full-val-epochs", type=int, default=52)
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)
    out_dir = artifacts_root / "evals" / snapshot_id / "pricing"
    table_path = out_dir / "pricing_training_table.parquet"

    # Match the schema fields used in existing `ft_transformer_tune_report.json`.
    report: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "table_path": str(table_path),
        "device": str(args.device),
        "policy_bound": float(args.policy_bound),
        "val_screen_rows": int(args.val_screen_rows),
        "top_k": int(args.top_k),
        "grid_screen_epochs": int(args.grid_screen_epochs),
        "grid_full_val_epochs": int(args.grid_full_val_epochs),
        "grid": [],
        "screen_results_sorted_by_val_rmse_clipped": [],
        "refine_results_sorted_by_val_rmse_clipped": [],
        "written_at_utc": _utc_now_iso(),
        "note": "FT-Transformer lane recreated as a placeholder entrypoint. "
        "If you want full reproduction, we can implement the training loop in copilot-v2.",
    }
    _write_json(out_dir / "ft_transformer_tune_report.json", report)

    raise RuntimeError(
        "FT-Transformer training code is not yet implemented in copilot-v2. "
        "I recreated the entrypoint and report schema, but you still need a PyTorch training "
        "implementation to reproduce the original metrics."
    )


if __name__ == "__main__":
    main()

