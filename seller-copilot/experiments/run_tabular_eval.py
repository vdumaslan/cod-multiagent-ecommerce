#!/usr/bin/env python3
"""
Tabular comparison: CatBoost vs sklearn HistGradientBoostingRegressor on price prediction.

Uses artifacts/splits/tabular_features.parquet.

Writes artifacts/evals/tabular/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from _paths import EVALS_DIR, SPLITS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["price", "log_price", "margin_pct"], default="log_price")
    args = parser.parse_args()

    df = pd.read_parquet(SPLITS_DIR / "tabular_features.parquet")
    df = df.dropna(subset=[args.target])

    cat_cols = [c for c in ["brand", "category", "subcategory"] if c in df.columns]
    exclude = {
        "product_id",
        "title",
        "description",
        "product_document",
        "split",
        "sku_id",
        "abc_velocity_class",
        "shelf_zone",
        "last_review_ts",
        "price",
        "log_price",
        "margin_pct",
    }
    exclude.add(args.target)
    num_cols = []
    for c in df.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    if args.target == "log_price" and "price" in num_cols:
        num_cols.remove("price")
    if args.target == "price" and "log_price" in num_cols:
        num_cols.remove("log_price")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[cat_cols + num_cols], train[args.target].astype(float)
    X_val, y_val = val[cat_cols + num_cols], val[args.target].astype(float)
    X_test, y_test = test[cat_cols + num_cols], test[args.target].astype(float)

    out = EVALS_DIR / "tabular"
    out.mkdir(parents=True, exist_ok=True)

    # CatBoost
    cat_idx = list(range(len(cat_cols)))
    cb_model = CatBoostRegressor(
        cat_features=cat_idx if cat_cols else None,
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40)
    pred_cb = cb_model.predict(X_test)
    metrics_cb = _metrics(y_test, pred_cb, "catboost")

    # HGB + ordinal categoricals
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    hgb = Pipeline(
        [
            ("prep", pre),
            (
                "est",
                HistGradientBoostingRegressor(
                    max_depth=8,
                    learning_rate=0.06,
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )
    hgb.fit(X_train, y_train)
    pred_h = hgb.predict(X_test)
    metrics_h = _metrics(y_test, pred_h, "hist_gradient_boosting")

    payload = {
        "target": args.target,
        "features": {"categorical": cat_cols, "numeric": num_cols},
        "test_rows": len(test),
        "catboost": metrics_cb,
        "sklearn_hgb": metrics_h,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _metrics(y_true: pd.Series, y_pred: np.ndarray, name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    if (y_true > 0).all() and (np.asarray(y_pred) > 0).all():
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        mape = None
    return {"model": name, "mae": float(mae), "rmse": float(rmse), "mape": mape}


if __name__ == "__main__":
    main()
