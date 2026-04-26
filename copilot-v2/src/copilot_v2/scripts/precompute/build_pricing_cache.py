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


def _read_registry(registry_path: Path) -> dict[str, Any]:
    return json.loads(registry_path.read_text(encoding="utf-8"))

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_tabpfn() -> Any:
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing TabPFN. Install `tabpfn` in the runtime that builds pricing caches.") from e
    return {"TabPFNRegressor": TabPFNRegressor}


def _build_cat_codebook(
    train_df: pd.DataFrame,
    *,
    cat_cols: list[str],
    n_fit: int,
) -> dict[str, list[str]]:
    """
    Reconstruct the categorical -> code mapping used in TabPFN training.

    The TabPFN tuning/training pipeline ordinal-encodes categoricals via:
      X[c] = X[c].astype("category").cat.codes
    on the *fit subset* (first n_fit train rows).

    To match that mapping at cache build time, we must reuse the exact category
    ordering induced by that fit subset.
    """
    fit = train_df.head(int(n_fit)).copy()
    codebook: dict[str, list[Any]] = {}
    for c in cat_cols:
        # IMPORTANT: match training behavior exactly:
        # training used `astype("category").cat.codes` on the raw column values,
        # which preserves NaN as missing (code -1). Do NOT cast to str here,
        # because that would convert NaN to the literal string "nan".
        s = fit[c]
        cats = list(pd.Categorical(s).categories.tolist())
        codebook[str(c)] = cats
    return codebook


def _prep_X(
    df: pd.DataFrame,
    *,
    cat_cols: list[str],
    num_cols: list[str],
    cat_codebook: dict[str, list[str]] | None = None,
) -> np.ndarray:
    # IMPORTANT: match the model's expected column order.
    # The saved TabPFN winner uses categorical_features_indices = [28,29,30],
    # which implies 28 numeric features first, then 3 categoricals at the end.
    X = df[num_cols + cat_cols].copy()
    # Match training ordinal encoding for categoricals.
    for c in cat_cols:
        s = X[c]
        if cat_codebook and str(c) in cat_codebook:
            s = pd.Categorical(s, categories=list(cat_codebook[str(c)]))
            X[c] = s.codes.astype(np.int32)
        else:
            X[c] = pd.Categorical(s).codes.astype(np.int32)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X.to_numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--registry-path", default="copilot-v2/artifacts/registry/registry.json")
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <artifacts_root>/caches/<snapshot_id>/pricing/",
    )
    p.add_argument("--write-json", action="store_true", help="Also write pricing_cache.json mapping product_id->float.")
    p.add_argument("--policy-bound", type=float, default=10.0, help="Clip predictions to +/- bound before writing cache.")
    p.add_argument("--predict-batch-size", type=int, default=2048)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="TabPFN inference device preference.")
    p.add_argument(
        "--table-path",
        default="",
        help="Override pricing_training_table.parquet path. Default: <artifacts_root>/evals/<snapshot_id>/pricing/pricing_training_table.parquet",
    )
    p.add_argument(
        "--state-path",
        default="",
        help="Override TabPFN state path. Default: registry pricing.model_artifact_path",
    )
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)
    registry = _read_registry(Path(args.registry_path))

    default_table = artifacts_root / "evals" / snapshot_id / "pricing" / "pricing_training_table.parquet"
    table_path = Path(args.table_path) if str(args.table_path).strip() else default_table
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing {table_path}. Build it with build_pricing_training_table.py first.")

    state_path = Path(args.state_path) if str(args.state_path).strip() else Path(str(registry["pricing"]["model_artifact_path"]))
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Missing TabPFN pricing model artifact at {state_path}. "
            "Rebuild the pricing model artifact first (or pass --state-path)."
        )

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (artifacts_root / "caches" / snapshot_id / "pricing")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(table_path)
    if "product_id" not in df.columns:
        raise RuntimeError("pricing_training_table.parquet missing product_id")

    TabPFNRegressor = _require_tabpfn()["TabPFNRegressor"]
    # Artifacts in this repo are saved via TabPFN's save_fit_state(); load with the official loader.
    if not hasattr(TabPFNRegressor, "load_from_fit_state"):
        raise RuntimeError("Your installed TabPFN is missing TabPFNRegressor.load_from_fit_state(). Please upgrade tabpfn.")
    # Prefer passing device to the loader (TabPFN supports this) to avoid CPU fallback.
    device_req = str(args.device).lower().strip()
    load_device: str = "cpu"
    if device_req == "cuda":
        try:
            import torch  # type: ignore

            load_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            load_device = "cpu"
    elif device_req == "cpu":
        load_device = "cpu"
    else:
        # auto
        try:
            import torch  # type: ignore

            load_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            load_device = "cpu"

    model = TabPFNRegressor.load_from_fit_state(state_path, device=load_device)

    # Use the exact feature schema recorded when the winner model was trained.
    train_report_path = Path(str(registry["pricing"].get("train_report_path") or ""))
    if not train_report_path.is_file():
        raise FileNotFoundError(
            f"Missing pricing train report at {train_report_path}. "
            "This report is required to reconstruct the exact feature schema for cache inference."
        )
    tr = _read_json(train_report_path)
    cat_cols = list(((tr.get("data") or {}).get("cat_cols") or []))
    num_cols = list(((tr.get("data") or {}).get("num_cols") or []))
    if not cat_cols or not num_cols:
        raise RuntimeError(f"Train report at {train_report_path} missing data.cat_cols or data.num_cols")
    # Handle occasional merge suffixes in the training table (e.g., subcategory_x/subcategory_y).
    col_map: dict[str, str] = {}
    for c in cat_cols + num_cols:
        if c in df.columns:
            col_map[c] = c
            continue
        if c == "subcategory":
            for alt in ["subcategory_x", "subcategory_y"]:
                if alt in df.columns:
                    col_map[c] = alt
                    break
        elif c == "category":
            for alt in ["category_x", "category_y"]:
                if alt in df.columns:
                    col_map[c] = alt
                    break
        elif c == "brand":
            for alt in ["brand_x", "brand_y"]:
                if alt in df.columns:
                    col_map[c] = alt
                    break
    missing = [c for c in (cat_cols + num_cols) if c not in col_map]
    if missing:
        raise RuntimeError(f"Pricing table at {table_path} missing required feature columns: {missing[:10]}")

    # Predict for all rows in the table (train/val/test/all) and aggregate to product_id.
    df2 = df.rename(columns={k: v for k, v in col_map.items() if k != v})
    # If we mapped expected->alt, rename alt back to expected to preserve column names for _prep_X.
    # Example: expected 'subcategory' -> actual 'subcategory_x' => rename subcategory_x -> subcategory.
    inv_map = {v: k for k, v in col_map.items()}
    df2 = df2.rename(columns=inv_map)

    # Build codebook from the same fit subset the winner used.
    n_train_fit = int(((tr.get("data") or {}).get("n_train_fit") or 0))
    # Fallback: use hyperparams.max_fit_rows if present.
    if n_train_fit <= 0:
        n_train_fit = int((((tr.get("hyperparams") or {}).get("max_fit_rows")) or 0))
    if n_train_fit <= 0:
        n_train_fit = 10_000

    if "split" not in df2.columns:
        raise RuntimeError("pricing_training_table.parquet missing split column; required to reconstruct TabPFN codebook.")
    train_df = df2[df2["split"].astype(str) == "train"].copy()
    cat_codebook = _build_cat_codebook(train_df, cat_cols=cat_cols, n_fit=int(n_train_fit))

    X_all = _prep_X(df2, cat_cols=cat_cols, num_cols=num_cols, cat_codebook=cat_codebook)
    preds: list[float] = []
    bs = max(1, int(args.predict_batch_size))
    for i in range(0, len(X_all), bs):
        chunk = X_all[i : i + bs]
        preds.extend(model.predict(chunk).astype(float).tolist())
    pred = np.asarray(preds, dtype=float)
    # Soft-cap instead of hard clipping to avoid boundary pileups at +/- policy_bound.
    b = float(args.policy_bound)
    pred = b * np.tanh(pred / b)

    out = pd.DataFrame({"product_id": df["product_id"].astype(str), "predicted_price_change_pct": pred})
    # If table contains multiple rows per product_id (should not, but safe), average.
    out = out.groupby("product_id", as_index=False).agg(predicted_price_change_pct=("predicted_price_change_pct", "mean"))

    out_path = out_dir / "pricing_cache.parquet"
    out.to_parquet(out_path, index=False)

    meta = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "snapshot_id": snapshot_id,
        "source_table": str(table_path),
        "model_state_path": str(state_path),
        "policy_bound": float(args.policy_bound),
        "rows": int(len(out)),
        "columns": list(out.columns),
    }
    (out_dir / "pricing_cache_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if bool(args.write_json):
        mapping = {str(r["product_id"]): float(r["predicted_price_change_pct"]) for _, r in out.iterrows()}
        (out_dir / "pricing_cache.json").write_text(json.dumps(mapping), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "parquet": str(out_path), "rows": int(len(out))}, indent=2))


if __name__ == "__main__":
    main()

