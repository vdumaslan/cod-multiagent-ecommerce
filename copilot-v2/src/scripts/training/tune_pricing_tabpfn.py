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


def _require_tabpfn() -> Any:
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing TabPFN. Install `tabpfn` to run this tuning lane.") from e
    return {"TabPFNRegressor": TabPFNRegressor}


def _prep_xy(df: pd.DataFrame, *, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    cat_cols = [c for c in ["brand", "category", "subcategory"] if c in df.columns]
    exclude = {
        "product_id",
        "title",
        "description",
        "product_document",
        "split",
        label_col,
    }
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[cat_cols + num_cols].copy()
    # TabPFN expects numerics; ordinal-encode categoricals.
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes.astype(np.int32)
    y = pd.to_numeric(df[label_col], errors="coerce").astype(float)
    return X, y


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
    p.add_argument("--device-requested", default="auto")
    p.add_argument("--full-max-fit-rows", type=int, default=0, help="0 means use full train rows")
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

    TabPFNRegressor = _require_tabpfn()["TabPFNRegressor"]

    grid = []
    for max_fit_rows in [4096, 8192]:
        for n_estimators in [4, 8]:
            for softmax_temperature in [0.85, 0.9, 1.0]:
                grid.append(
                    {
                        "max_fit_rows": int(max_fit_rows),
                        "n_estimators": int(n_estimators),
                        "softmax_temperature": float(softmax_temperature),
                    }
                )

    X_train, y_train = _prep_xy(train, label_col=label_col)
    X_val_s, y_val_s = _prep_xy(val_screen, label_col=label_col)

    screen_results = []
    for cfg in grid:
        t0 = time.time()
        # Fit subset to cap runtime.
        n_fit = min(int(cfg["max_fit_rows"]), len(X_train))
        X_fit = X_train.iloc[:n_fit].to_numpy()
        y_fit = y_train.iloc[:n_fit].to_numpy()
        model = TabPFNRegressor(
            n_estimators=int(cfg["n_estimators"]),
            softmax_temperature=float(cfg["softmax_temperature"]),
            device=args.device_requested,
        )
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val_s.to_numpy()).astype(float)
        seconds_fit = time.time() - t0

        y_true = y_val_s.to_numpy(dtype=float)
        pred_clip = _clip(pred, float(args.policy_bound))
        y_clip = _clip(y_true, float(args.policy_bound))
        violation = float(np.mean(np.abs(pred) > float(args.policy_bound)))

        screen_results.append(
            {
                "cfg_id": _cfg_id(cfg),
                **cfg,
                "n_rows": int(len(y_true)),
                "rmse_unclipped": _rmse(y_true, pred),
                "mae_unclipped": _mae(y_true, pred),
                "rmse_clipped": _rmse(y_clip, pred_clip),
                "mae_clipped": _mae(y_clip, pred_clip),
                "violation_rate_unclipped": violation,
                "violation_rate_clipped": 0.0,
                "seconds_fit": float(seconds_fit),
            }
        )

    screen_results = sorted(screen_results, key=lambda r: r["rmse_clipped"])
    top = screen_results[: int(args.top_k)]

    # Full-val refine on top-k
    X_val, y_val = _prep_xy(val, label_col=label_col)
    X_full_train, y_full_train = _prep_xy(train, label_col=label_col)

    full_max_fit_rows_effective = int(args.full_max_fit_rows) if int(args.full_max_fit_rows) > 0 else int(len(train))

    refine_results = []
    for r in top:
        cfg = {k: r[k] for k in ["max_fit_rows", "n_estimators", "softmax_temperature"]}
        t0 = time.time()
        n_fit = min(int(full_max_fit_rows_effective), len(X_full_train))
        model = TabPFNRegressor(
            n_estimators=int(cfg["n_estimators"]),
            softmax_temperature=float(cfg["softmax_temperature"]),
            device=args.device_requested,
        )
        model.fit(X_full_train.iloc[:n_fit].to_numpy(), y_full_train.iloc[:n_fit].to_numpy())
        pred = model.predict(X_val.to_numpy()).astype(float)
        seconds_fit = time.time() - t0

        y_true = y_val.to_numpy(dtype=float)
        pred_clip = _clip(pred, float(args.policy_bound))
        y_clip = _clip(y_true, float(args.policy_bound))
        violation = float(np.mean(np.abs(pred) > float(args.policy_bound)))
        refine_results.append(
            {
                "cfg_id": _cfg_id(cfg),
                **cfg,
                "n_rows": int(len(y_true)),
                "rmse_unclipped": _rmse(y_true, pred),
                "mae_unclipped": _mae(y_true, pred),
                "rmse_clipped": _rmse(y_clip, pred_clip),
                "mae_clipped": _mae(y_clip, pred_clip),
                "violation_rate_unclipped": violation,
                "violation_rate_clipped": 0.0,
                "seconds_fit": float(seconds_fit),
            }
        )

    refine_results = sorted(refine_results, key=lambda rr: rr["rmse_clipped"])

    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "table_path": str(table_path),
        "device_requested": str(args.device_requested),
        "policy_bound": float(args.policy_bound),
        "val_screen_rows": int(args.val_screen_rows),
        "top_k": int(args.top_k),
        "full_max_fit_rows_cli": int(args.full_max_fit_rows),
        "full_max_fit_rows_effective": int(full_max_fit_rows_effective),
        "grid": grid,
        "screen_results_sorted_by_val_rmse_clipped": screen_results,
        "refine_results_sorted_by_val_rmse_clipped": refine_results,
        "written_at_utc": _utc_now_iso(),
    }
    _write_json(out_dir / "tabpfn_tune_report.json", report)
    print(json.dumps({"ok": True, "out": str(out_dir / "tabpfn_tune_report.json")}, indent=2))


if __name__ == "__main__":
    main()

