from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _ranking_metrics(df: pd.DataFrame, score_col: str, k: int = 10) -> dict[str, float]:
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []

    for _, g in df.groupby("query_text"):
        g = g.sort_values(score_col, ascending=False)
        labels = g["relevance_label"].astype(int).to_numpy()
        if labels.sum() == 0:
            continue
        top = labels[:k]
        recalls.append(float(1.0 if top.sum() > 0 else 0.0))
        precisions.append(float(top.mean()))
        pos_idx = np.where(top == 1)[0]
        mrrs.append(float(1.0 / (pos_idx[0] + 1)) if len(pos_idx) else 0.0)

        # Simple DCG/IDCG for binary relevance
        gains = top / np.log2(np.arange(2, len(top) + 2))
        dcg = float(gains.sum())
        ideal = np.sort(labels)[::-1][:k]
        idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="seller-copilot/artifacts")
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/demo1_evidence")
    parser.add_argument("--ranking-train-max", type=int, default=20000)
    parser.add_argument("--ranking-test-max", type=int, default=5000)
    parser.add_argument("--pricing-train-max", type=int, default=80000)
    parser.add_argument("--pricing-test-max", type=int, default=12000)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_prep = json.loads((root / "data" / "data_prep_summary.json").read_text(encoding="utf-8"))
    data_analytics = json.loads((root / "data_analytics_summary.json").read_text(encoding="utf-8"))
    sentiment = json.loads((root / "sentiment" / "classification_report.json").read_text(encoding="utf-8"))

    ranking_train = pd.read_parquet(root / "data" / "ranking_train.parquet")
    ranking_test = pd.read_parquet(root / "data" / "ranking_test.parquet")
    if len(ranking_train) > args.ranking_train_max:
        ranking_train = ranking_train.sample(args.ranking_train_max, random_state=42).reset_index(drop=True)
    if len(ranking_test) > args.ranking_test_max:
        ranking_test = ranking_test.sample(args.ranking_test_max, random_state=42).reset_index(drop=True)

    ranking_train = ranking_train.dropna(subset=["query_text", "product_document", "relevance_label"])
    ranking_test = ranking_test.dropna(subset=["query_text", "product_document", "relevance_label"])
    x_train = (ranking_train["query_text"].astype(str) + " [SEP] " + ranking_train["product_document"].astype(str)).tolist()
    y_train = ranking_train["relevance_label"].astype(int).to_numpy()
    x_test = (ranking_test["query_text"].astype(str) + " [SEP] " + ranking_test["product_document"].astype(str)).tolist()

    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2")
    clf = SGDClassifier(loss="log_loss", max_iter=8, tol=1e-3, random_state=42)
    clf.fit(vectorizer.transform(x_train), y_train)
    prob = clf.predict_proba(vectorizer.transform(x_test))[:, 1]
    ranking_eval = ranking_test.copy()
    ranking_eval["score"] = prob
    pairwise_acc = float(((ranking_eval["score"] >= 0.5).astype(int) == ranking_eval["relevance_label"].astype(int)).mean())
    ranking_metrics = _ranking_metrics(ranking_eval, "score", k=10)

    pricing_train = pd.read_parquet(root / "data" / "pricing_train.parquet")
    pricing_test = pd.read_parquet(root / "data" / "pricing_test.parquet")
    if len(pricing_train) > args.pricing_train_max:
        pricing_train = pricing_train.sample(args.pricing_train_max, random_state=42).reset_index(drop=True)
    if len(pricing_test) > args.pricing_test_max:
        pricing_test = pricing_test.sample(args.pricing_test_max, random_state=42).reset_index(drop=True)

    feature_cols = ["price", "avg_rating", "review_count", "positive_ratio", "rating_price_ratio"]
    target_col = "target_price"
    pricing_train = pricing_train.dropna(subset=feature_cols + [target_col])
    pricing_test = pricing_test.dropna(subset=feature_cols + [target_col])

    reg = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_depth=6, random_state=42))
    reg.fit(pricing_train[feature_cols], pricing_train[target_col])
    pred = reg.predict(pricing_test[feature_cols])
    y = pricing_test[target_col].to_numpy()
    pricing_metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.clip(y, 1e-3, None))) * 100.0),
    }

    ongoing_ml = {
        "sentiment": {
            "model": sentiment.get("model_id"),
            "num_eval_rows": sentiment.get("num_eval_rows"),
            "accuracy": sentiment.get("metrics", {}).get("accuracy"),
            "macro_f1": sentiment.get("metrics", {}).get("macro avg", {}).get("f1-score"),
            "weighted_f1": sentiment.get("metrics", {}).get("weighted avg", {}).get("f1-score"),
        },
        "ranking_baseline": {
            "model": "HashingVectorizer + SGDClassifier baseline",
            "num_train_pairs": int(len(ranking_train)),
            "num_eval_pairs": int(len(ranking_eval)),
            "pairwise_accuracy": pairwise_acc,
            **ranking_metrics,
        },
        "pricing_baseline": {
            "model": "HistGradientBoostingRegressor baseline",
            "num_train_rows": int(len(pricing_train)),
            "num_eval_rows": int(len(pricing_test)),
            **pricing_metrics,
        },
    }

    (out_dir / "training_test_data_preparation.json").write_text(json.dumps(data_prep, indent=2), encoding="utf-8")
    (out_dir / "data_analytics_results.json").write_text(json.dumps(data_analytics, indent=2), encoding="utf-8")
    (out_dir / "ongoing_ml_results.json").write_text(json.dumps(ongoing_ml, indent=2), encoding="utf-8")

    summary_md = f"""# Demo 1 Evidence Pack

- Data preprocessing and preparation: `training_test_data_preparation.json`
- Data analytics results: `data_analytics_results.json`
- Ongoing ML results: `ongoing_ml_results.json`

## Headline Metrics
- Sentiment macro F1: `{ongoing_ml['sentiment']['macro_f1']:.4f}`
- Ranking nDCG@10: `{ongoing_ml['ranking_baseline']['ndcg@10']:.4f}`
- Ranking MRR@10: `{ongoing_ml['ranking_baseline']['mrr@10']:.4f}`
- Pricing MAE: `{ongoing_ml['pricing_baseline']['mae']:.4f}`
- Pricing RMSE: `{ongoing_ml['pricing_baseline']['rmse']:.4f}`
"""
    (out_dir / "demo1_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Saved evidence pack to {out_dir}")


if __name__ == "__main__":
    main()
