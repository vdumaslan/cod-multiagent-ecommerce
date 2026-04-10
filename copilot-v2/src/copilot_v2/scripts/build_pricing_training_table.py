from __future__ import annotations

import argparse
import json
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


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _label_recipe() -> dict[str, Any]:
    # Mirrors `pricing_label_recipe.json` already in artifacts (v1.1.0).
    return {
        "schema_version": 1,
        "label_recipe_version": "1.1.0",
        "label_column": "recommended_price_change_pct",
        "grid": {"pct_min": -10, "pct_max": 10, "step": 1, "n_candidates": 21},
        "formula": {
            "elasticity": -1.25,
            "unit_cost_fraction_of_price": 0.62,
            "revenue_weight": 0.25,
            "contribution_weight": 0.65,
            "clearance_weight": 0.1,
            "reference_price_anchor_weight": 0.45,
            "delta_risk_penalty": 0.18,
            "p_ref_definition": "subcategory_median_price if positive, else current price",
            "u0_definition": "max(total_units_sold / max(n_sale_days,1), mean_daily_revenue/price, 1e-4)",
            "clearance_definition": "overstock_ratio * max(0, -delta_frac) * Q * p_new",
        },
    }


def _compute_labels(df: pd.DataFrame, *, recipe: dict[str, Any], seed: int) -> pd.Series:
    # Deterministic label generator: grid search on price change percent.
    pct_min = int(recipe["grid"]["pct_min"])
    pct_max = int(recipe["grid"]["pct_max"])
    step = int(recipe["grid"]["step"])
    grid = np.arange(pct_min, pct_max + 1, step, dtype=np.float32)

    elasticity = float(recipe["formula"]["elasticity"])
    unit_cost_frac = float(recipe["formula"]["unit_cost_fraction_of_price"])
    w_rev = float(recipe["formula"]["revenue_weight"])
    w_contrib = float(recipe["formula"]["contribution_weight"])
    w_clear = float(recipe["formula"]["clearance_weight"])
    w_anchor = float(recipe["formula"]["reference_price_anchor_weight"])
    w_risk = float(recipe["formula"]["delta_risk_penalty"])

    price = _safe_num(df["price"]).to_numpy(np.float32)
    mean_daily_rev = _safe_num(df.get("mean_daily_revenue", 0.0)).fillna(0.0).to_numpy(np.float32)
    total_units = _safe_num(df.get("total_units_sold", 0.0)).fillna(0.0).to_numpy(np.float32)
    n_sale_days = _safe_num(df.get("n_sale_days", 1.0)).fillna(1.0).to_numpy(np.float32)
    total_returns = _safe_num(df.get("total_returns", 0.0)).fillna(0.0).to_numpy(np.float32)
    on_hand = _safe_num(df.get("on_hand_units", 0.0)).fillna(0.0).to_numpy(np.float32)
    safety = _safe_num(df.get("safety_stock_units", 0.0)).fillna(0.0).to_numpy(np.float32)

    # Reference price: subcategory median if available, else current price.
    p_ref = _safe_num(df.get("subcategory_median_price", np.nan)).to_numpy(np.float32)
    p_ref = np.where(np.isfinite(p_ref) & (p_ref > 0), p_ref, price)

    # Baseline demand proxy.
    u0 = np.maximum(total_units / np.maximum(n_sale_days, 1.0), mean_daily_rev / np.maximum(price, 1e-6))
    u0 = np.maximum(u0, 1e-4).astype(np.float32)

    # Overstock ratio: uses sellable inventory estimate.
    available = np.maximum(on_hand - safety, 0.0).astype(np.float32)
    overstock_ratio = available / np.maximum(1.0, (u0 * 30.0)).astype(np.float32)

    unit_cost = unit_cost_frac * price

    # Evaluate all candidates vectorized: N x G
    delta = (grid / 100.0).reshape(1, -1)  # (1,G)
    p_new = price.reshape(-1, 1) * (1.0 + delta)
    p_new = np.maximum(p_new, 1e-3)

    # Demand response: Q = u0 * (p_new/price)^elasticity
    ratio = p_new / np.maximum(price.reshape(-1, 1), 1e-6)
    Q = u0.reshape(-1, 1) * (ratio**elasticity)
    Q = np.maximum(Q, 1e-6)

    revenue = Q * p_new
    contrib = Q * (p_new - unit_cost.reshape(-1, 1))

    # Clearance term: only for markdown (negative delta).
    clearance = overstock_ratio.reshape(-1, 1) * np.maximum(0.0, -delta) * revenue

    # Anchor to reference price (penalize large deviation).
    anchor_pen = np.abs(p_new - p_ref.reshape(-1, 1)) / np.maximum(p_ref.reshape(-1, 1), 1e-6)

    # Returns penalty: discourage strong markdown if high returns.
    returns_rate = total_returns.reshape(-1, 1) / np.maximum(1.0, total_units.reshape(-1, 1))
    returns_pen = returns_rate * np.maximum(0.0, -delta) * revenue

    risk_pen = w_risk * np.abs(delta) * revenue

    score = (
        w_rev * revenue
        + w_contrib * contrib
        + w_clear * clearance
        - w_anchor * anchor_pen * revenue
        - 0.05 * returns_pen
        - risk_pen
    )

    best_idx = score.argmax(axis=1)
    best_pct = grid[best_idx].astype(np.float32)
    return pd.Series(best_pct, index=df.index, name="recommended_price_change_pct")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--max-rows", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)

    tabular_path = artifacts_root / "features" / snapshot_id / "tabular_features.parquet"
    if not tabular_path.exists():
        raise FileNotFoundError(f"Missing {tabular_path}")

    out_dir = artifacts_root / "evals" / snapshot_id / "pricing"
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe = _label_recipe()
    _write_json(out_dir / "pricing_label_recipe.json", {**recipe, "n_rows_input": int(args.max_rows), "written_at_utc": _utc_now_iso()})

    df = pd.read_parquet(tabular_path)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)

    # Build missing helper if not present.
    if "n_sale_days" not in df.columns and "total_units_sold" in df.columns:
        # If sales_daily was aggregated elsewhere, we may not have n_sale_days; default to 30.
        df["n_sale_days"] = 30

    labels = _compute_labels(df, recipe=recipe, seed=int(args.seed))
    labels_df = pd.DataFrame({"product_id": df["product_id"].astype(str), "recommended_price_change_pct": labels})
    labels_path = out_dir / "pricing_labels.parquet"
    labels_df.to_parquet(labels_path, index=False)

    # Merge to training table, filter valid price and valid label.
    merged = df.merge(labels_df, on="product_id", how="inner")
    price_num = _safe_num(merged["price"])
    label_num = _safe_num(merged["recommended_price_change_pct"])
    mask = price_num.notna() & (price_num > 0) & label_num.notna() & np.isfinite(label_num.to_numpy())
    merged = merged.loc[mask].copy()

    out_parquet = out_dir / "pricing_training_table.parquet"
    merged.to_parquet(out_parquet, index=False)

    # Report (schema_version=1) similar to existing pricing_training_report.json
    def _bins_top(s: pd.Series, top_n: int = 25) -> dict[str, int]:
        vc = s.round(0).astype(int).value_counts().head(top_n)
        return {str(k): int(v) for k, v in vc.items()}

    report = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "tabular_path": str(tabular_path),
        "labels_path": str(labels_path),
        "out_parquet": str(out_parquet),
        "label_col": "recommended_price_change_pct",
        "filters": {
            "valid_price_numeric_gt_0": True,
            "valid_label_not_null": True,
            "price_and_label_inner_logic": "price_num finite & >0 AND label not null & finite",
        },
        "counts_before": {"tabular_rows": int(len(df)), "labels_rows": int(len(labels_df))},
        "counts_after": {
            "merged_rows": int(len(merged)),
            "split_counts": {k: int((merged.get("split") == k).sum()) for k in ["train", "val", "test"] if "split" in merged.columns},
            "n_label_distinct_rounded_bins": int(merged["recommended_price_change_pct"].round(0).nunique()),
        },
        "label_bins_top_25": _bins_top(merged["recommended_price_change_pct"], top_n=25),
        "per_split": {},
        "warnings": [],
        "written_at_utc": _utc_now_iso(),
    }
    if "split" in merged.columns:
        for sp in ["train", "val", "test"]:
            part = merged[merged["split"] == sp]
            if part.empty:
                continue
            report["per_split"][sp] = {
                "n_rows": int(len(part)),
                "label_mean": float(part["recommended_price_change_pct"].mean()),
                "label_std": float(part["recommended_price_change_pct"].std(ddof=0)),
                "label_min": float(part["recommended_price_change_pct"].min()),
                "label_max": float(part["recommended_price_change_pct"].max()),
                "label_bins_top": _bins_top(part["recommended_price_change_pct"], top_n=18),
            }

    _write_json(out_dir / "pricing_training_report.json", report)

    print(json.dumps({"ok": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

