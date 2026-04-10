from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cfg_id(cfg: dict[str, Any]) -> str:
    b = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:10]


def _require_catboost() -> Any:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing CatBoost. Install `catboost` to run this tuning lane.") from e
    return {"CatBoostRegressor": CatBoostRegressor}


def _clip(y: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(y, -bound, bound)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2) ** 0.5)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--policy-bound", type=float, default=10.0)
    p.add_argument("--val-screen-rows", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)
    out_dir = artifacts_root / "evals" / snapshot_id / "pricing"
    table_path = out_dir / "pricing_training_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing {table_path}. Run build_pricing_training_table.py first.")

    df = pd.read_parquet(table_path)
    label_col = "recommended_price_change_pct"
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    if len(val) > int(args.val_screen_rows) and int(args.val_screen_rows) > 0:
        val_screen = val.sample(int(args.val_screen_rows), random_state=42).reset_index(drop=True)
    else:
        val_screen = val

    cat_cols = [c for c in ["brand", "category", "subcategory"] if c in df.columns]
    exclude = {"product_id", "title", "description", "product_document", "split", label_col}
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]

    X_train = train[cat_cols + num_cols]
    y_train = pd.to_numeric(train[label_col], errors="coerce").astype(float)
    X_val_s = val_screen[cat_cols + num_cols]
    y_val_s = pd.to_numeric(val_screen[label_col], errors="coerce").astype(float)

    CatBoostRegressor = _require_catboost()["CatBoostRegressor"]

    grid = []
    for depth in [4, 6, 8]:
        for lr in [0.03, 0.05]:
            for l2 in [1.0, 3.0]:
                grid.append({"depth": int(depth), "learning_rate": float(lr), "l2_leaf_reg": float(l2), "iterations": 800})

    screen_results = []
    for cfg in grid:
        t0 = time.time()
        model = CatBoostRegressor(
            depth=int(cfg["depth"]),
            learning_rate=float(cfg["learning_rate"]),
            l2_leaf_reg=float(cfg["l2_leaf_reg"]),
            iterations=int(cfg["iterations"]),
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train, cat_features=list(range(len(cat_cols))) if cat_cols else None)
        pred = model.predict(X_val_s).astype(float)
        seconds_fit = time.time() - t0

        y_true = y_val_s.to_numpy(dtype=float)
        pred_clip = _clip(pred, float(args.policy_bound))
        y_clip = _clip(y_true, float(args.policy_bound))
        violation = float(np.mean(np.abs(pred) > float(args.policy_bound)))

        screen_results.append(
            {
                "cfg_id": _cfg_id(cfg),
                "val_rmse_unclipped": _rmse(y_true, pred),
                "val_mae_unclipped": _mae(y_true, pred),
                "val_rmse_clipped": _rmse(y_clip, pred_clip),
                "val_mae_clipped": _mae(y_clip, pred_clip),
                "violation_rate_unclipped": violation,
                "violation_rate_clipped": 0.0,
                "seconds_fit": float(seconds_fit),
                "best_iteration": int(cfg["iterations"]),
                "cfg": cfg,
            }
        )

    screen_results = sorted(screen_results, key=lambda r: r["val_rmse_clipped"])
    top = screen_results[: int(args.top_k)]

    # Refine: re-fit top-k on full val (same simple approach as reports)
    X_val = val[cat_cols + num_cols]
    y_val = pd.to_numeric(val[label_col], errors="coerce").astype(float).to_numpy(dtype=float)
    refine_results = []
    for r in top:
        cfg = r["cfg"]
        t0 = time.time()
        model = CatBoostRegressor(
            depth=int(cfg["depth"]),
            learning_rate=float(cfg["learning_rate"]),
            l2_leaf_reg=float(cfg["l2_leaf_reg"]),
            iterations=int(cfg["iterations"]),
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train, cat_features=list(range(len(cat_cols))) if cat_cols else None)
        pred = model.predict(X_val).astype(float)
        seconds_fit = time.time() - t0

        pred_clip = _clip(pred, float(args.policy_bound))
        y_clip = _clip(y_val, float(args.policy_bound))
        violation = float(np.mean(np.abs(pred) > float(args.policy_bound)))
        refine_results.append(
            {
                "cfg_id": _cfg_id(cfg),
                "val_rmse_unclipped": _rmse(y_val, pred),
                "val_mae_unclipped": _mae(y_val, pred),
                "val_rmse_clipped": _rmse(y_clip, pred_clip),
                "val_mae_clipped": _mae(y_clip, pred_clip),
                "violation_rate_unclipped": violation,
                "violation_rate_clipped": 0.0,
                "seconds_fit": float(seconds_fit),
                "best_iteration": int(cfg["iterations"]),
                "cfg": cfg,
            }
        )
    refine_results = sorted(refine_results, key=lambda rr: rr["val_rmse_clipped"])

    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "table_path": str(table_path),
        "label_col": label_col,
        "policy_bound": float(args.policy_bound),
        "cat_cols": cat_cols,
        "grid": grid,
        "val_screen_rows": int(args.val_screen_rows),
        "top_k": int(args.top_k),
        "screen_results_sorted_by_val_rmse_clipped": screen_results,
        "refine_results_sorted_by_val_rmse_clipped": refine_results,
        "written_at_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "catboost_tune_report.json", report)
    print(json.dumps({"ok": True, "out": str(out_dir / "catboost_tune_report.json")}, indent=2))


if __name__ == "__main__":
    main()

