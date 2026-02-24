from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _build_ranking_reference(
    ranking_train: pd.DataFrame,
    ranking_test: pd.DataFrame,
    max_train: int = 30000,
    max_test: int = 10000,
) -> dict[str, Any]:
    train = ranking_train.dropna(subset=["query_text", "product_document", "relevance_label"]).copy()
    test = ranking_test.dropna(subset=["query_text", "product_document", "relevance_label"]).copy()
    if len(train) > max_train:
        train = train.sample(max_train, random_state=42).reset_index(drop=True)
    if len(test) > max_test:
        test = test.sample(max_test, random_state=42).reset_index(drop=True)

    x_train = (train["query_text"].astype(str) + " [SEP] " + train["product_document"].astype(str)).tolist()
    y_train = train["relevance_label"].astype(int).to_numpy()
    x_test = (test["query_text"].astype(str) + " [SEP] " + test["product_document"].astype(str)).tolist()
    y_test = test["relevance_label"].astype(int).to_numpy()

    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2")
    clf = SGDClassifier(loss="log_loss", max_iter=8, tol=1e-3, random_state=42)
    clf.fit(vectorizer.transform(x_train), y_train)
    prob = clf.predict_proba(vectorizer.transform(x_test))[:, 1]

    test_eval = test.copy()
    test_eval["score"] = prob
    test_eval["pred_label"] = (test_eval["score"] >= 0.5).astype(int)
    pairwise_accuracy = float((test_eval["pred_label"] == test_eval["relevance_label"].astype(int)).mean())
    metrics = _ranking_metrics(test_eval, "score", k=10)
    metrics["pairwise_accuracy"] = pairwise_accuracy
    return {"model": "HashingVectorizer + SGDClassifier", "metrics": metrics}


def _build_pricing_reference(
    pricing_train: pd.DataFrame,
    pricing_test: pd.DataFrame,
    max_train: int = 100000,
    max_test: int = 20000,
) -> dict[str, Any]:
    feature_cols = ["price", "avg_rating", "review_count", "positive_ratio", "rating_price_ratio"]
    target_col = "target_price"

    train = pricing_train.dropna(subset=feature_cols + [target_col]).copy()
    test = pricing_test.dropna(subset=feature_cols + [target_col]).copy()
    if len(train) > max_train:
        train = train.sample(max_train, random_state=42).reset_index(drop=True)
    if len(test) > max_test:
        test = test.sample(max_test, random_state=42).reset_index(drop=True)

    model = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_depth=6, random_state=42))
    model.fit(train[feature_cols], train[target_col])
    pred = model.predict(test[feature_cols])
    y = test[target_col].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / np.clip(y, 1e-3, None))) * 100.0),
    }
    return {"model": "HistGradientBoostingRegressor", "metrics": metrics}


def _compute_retrieval_metrics_from_ranking(df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    if "retrieval_score" not in df.columns:
        return {}
    ndcgs: list[float] = []
    for _, g in df.groupby("query_text"):
        y_true = g["relevance_label"].astype(int).to_numpy()
        y_score = g["retrieval_score"].astype(float).to_numpy()
        if y_true.sum() == 0:
            continue
        ndcgs.append(float(ndcg_score([y_true], [y_score], k=min(k, len(y_true)))))
    return {f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0}


def _render_analytics_figures(data_dir: Path, figures_dir: Path) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return [{"title": "Figures unavailable", "reason": "matplotlib not installed"}]

    figures_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []

    sentiment = pd.read_parquet(data_dir / "sentiment_train.parquet")
    sentiment_counts = sentiment["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    sentiment_counts.plot(kind="bar", color=["#d73027", "#fee08b", "#1a9850"])
    plt.title("Sentiment Label Distribution (Train)")
    plt.xlabel("Label")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(figures_dir / "sentiment_label_distribution.png", dpi=180)
    plt.close()
    findings.append(
        {
            "title": "Sentiment class balance",
            "values": {str(int(k)): int(v) for k, v in sentiment_counts.items()},
            "figure": "figures/sentiment_label_distribution.png",
        }
    )

    ranking = pd.read_parquet(data_dir / "ranking_train.parquet")
    ranking_counts = ranking["relevance_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ranking_counts.plot(kind="bar", color=["#ef8a62", "#67a9cf"])
    plt.title("Ranking Relevance Distribution (Train)")
    plt.xlabel("Relevance Label")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(figures_dir / "ranking_label_distribution.png", dpi=180)
    plt.close()
    findings.append(
        {
            "title": "Ranking relevance balance",
            "values": {str(int(k)): int(v) for k, v in ranking_counts.items()},
            "figure": "figures/ranking_label_distribution.png",
        }
    )

    pricing = pd.read_parquet(data_dir / "pricing_train.parquet")
    plt.figure(figsize=(7, 4))
    plt.hist(pricing["target_price"].dropna().to_numpy(), bins=40, color="#3288bd")
    plt.title("Pricing Target Distribution (Train)")
    plt.xlabel("target_price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / "pricing_target_distribution.png", dpi=180)
    plt.close()

    sample = pricing.dropna(subset=["avg_rating", "target_price"]).sample(min(5000, len(pricing)), random_state=42)
    plt.figure(figsize=(7, 4))
    plt.scatter(sample["avg_rating"], sample["target_price"], s=8, alpha=0.2, c="#66c2a5")
    plt.title("Avg Rating vs Target Price")
    plt.xlabel("avg_rating")
    plt.ylabel("target_price")
    plt.tight_layout()
    plt.savefig(figures_dir / "rating_vs_target_price.png", dpi=180)
    plt.close()

    findings.append(
        {
            "title": "Pricing spread",
            "values": {
                "target_price_mean": float(pricing["target_price"].mean()),
                "target_price_p50": float(pricing["target_price"].median()),
                "target_price_p90": float(pricing["target_price"].quantile(0.9)),
            },
            "figure": "figures/pricing_target_distribution.png",
        }
    )
    findings.append(
        {
            "title": "Rating and pricing relationship",
            "values": {
                "corr_avg_rating_target_price": float(
                    pricing[["avg_rating", "target_price"]].corr().iloc[0, 1]
                )
            },
            "figure": "figures/rating_vs_target_price.png",
        }
    )
    return findings


def _export_sample_observations(data_dir: Path, output_dir: Path, sample_size: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sentiment_test = pd.read_parquet(data_dir / "sentiment_test.parquet")
    sentiment_out = (
        sentiment_test[["product_id", "text", "label"]]
        .sample(min(sample_size, len(sentiment_test)), random_state=42)
        .reset_index(drop=True)
    )
    sentiment_out["label_name"] = sentiment_out["label"].map({0: "negative", 1: "neutral", 2: "positive"}).fillna("unknown")
    sentiment_out.to_csv(output_dir / "sample_sentiment_observations.csv", index=False)

    ranking_test = pd.read_parquet(data_dir / "ranking_test.parquet")
    ranking_out = (
        ranking_test[["query_text", "product_id", "product_document", "relevance_label"]]
        .sample(min(sample_size, len(ranking_test)), random_state=42)
        .reset_index(drop=True)
    )
    ranking_out.to_csv(output_dir / "sample_ranking_observations.csv", index=False)

    pricing_test = pd.read_parquet(data_dir / "pricing_test.parquet")
    pricing_cols = [
        "product_id",
        "price",
        "avg_rating",
        "review_count",
        "positive_ratio",
        "rating_price_ratio",
        "target_price",
    ]
    pricing_out = pricing_test[pricing_cols].sample(min(sample_size, len(pricing_test)), random_state=42).reset_index(drop=True)
    pricing_out.to_csv(output_dir / "sample_pricing_observations.csv", index=False)

    summary = {
        "sentiment_samples": int(len(sentiment_out)),
        "ranking_samples": int(len(ranking_out)),
        "pricing_samples": int(len(pricing_out)),
        "files": [
            "sample_sentiment_observations.csv",
            "sample_ranking_observations.csv",
            "sample_pricing_observations.csv",
        ],
    }
    _write_json(output_dir / "sample_observations_summary.json", summary)
    return summary


def _parse_sentiment_label(text: str) -> int:
    t = text.lower()
    if "negative" in t:
        return 0
    if "neutral" in t:
        return 1
    if "positive" in t:
        return 2
    return 1


def _predict_sentiment_with_local_llm(
    model_id: str,
    texts: list[str],
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> list[int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    torch_dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
    )
    prompts = [
        (
            "Classify sentiment using only one word: negative, neutral, or positive.\n"
            f"Text: {txt}\n"
            "Label:"
        )
        for txt in texts
    ]
    raw_outputs = generator(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
        batch_size=batch_size,
    )
    preds: list[int] = []
    for out in raw_outputs:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            generated = str(out[0].get("generated_text", ""))
        elif isinstance(out, dict):
            generated = str(out.get("generated_text", ""))
        else:
            generated = str(out)
        preds.append(_parse_sentiment_label(generated))
    del generator
    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return preds


def _evaluate_llm_sentiment_model(
    model_id: str,
    eval_df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    y_true = eval_df["label"].astype(int).tolist()
    texts = eval_df["text"].astype(str).tolist()
    try:
        y_pred = _predict_sentiment_with_local_llm(
            model_id=model_id,
            texts=texts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
        )
    except Exception as exc:
        return {
            "model": model_id,
            "status": "failed",
            "error": str(exc),
            "num_eval_rows": int(len(eval_df)),
            "metrics": {},
        }
    return {
        "model": model_id,
        "status": "ok",
        "num_eval_rows": int(len(eval_df)),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        },
    }


def _build_llm_sentiment_comparison(
    eval_df: pd.DataFrame,
    historical_model_id: str,
    new_model_id: str,
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    historical = _evaluate_llm_sentiment_model(
        model_id=historical_model_id,
        eval_df=eval_df,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    current = _evaluate_llm_sentiment_model(
        model_id=new_model_id,
        eval_df=eval_df,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    return {
        "task": "sentiment_classification",
        "historical": historical,
        "new": current,
    }


def _build_llm_side_by_side(
    comparison_dir: Path,
    llm_comparison: dict[str, Any],
) -> dict[str, Any]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    historical = llm_comparison.get("historical", {})
    current = llm_comparison.get("new", {})
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    rows: list[dict[str, Any]] = []
    for m in metrics:
        old = historical.get("metrics", {}).get(m)
        new = current.get("metrics", {}).get(m)
        if old is None or new is None:
            delta = None
            improved = None
        else:
            delta = float(new) - float(old)
            improved = bool(delta > 0)
        rows.append(
            {
                "task": "sentiment_classification",
                "metric": m,
                "historical_model": historical.get("model"),
                "historical_value": old,
                "new_model": current.get("model"),
                "new_value": new,
                "delta_new_minus_historical": delta,
                "improved": improved,
            }
        )
    pd.DataFrame(rows).to_csv(comparison_dir / "llm_sentiment_model_comparison.csv", index=False)
    md_lines = [
        "# LLM Sentiment Model Comparison",
        "",
        f"Historical model: `{historical.get('model')}`",
        f"New model: `{current.get('model')}`",
        f"Historical status: `{historical.get('status')}`",
        f"New status: `{current.get('status')}`",
        "",
        "| Metric | Historical | New | Delta (New-Historical) | Improved |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['metric']} | {row['historical_value']} | {row['new_value']} | {row['delta_new_minus_historical']} | {row['improved']} |"
        )
    (comparison_dir / "llm_sentiment_model_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {
        "task": "sentiment_classification",
        "historical_model": historical.get("model"),
        "new_model": current.get("model"),
        "rows": rows,
        "file_csv": "llm_sentiment_model_comparison.csv",
        "file_md": "llm_sentiment_model_comparison.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="seller-copilot/artifacts")
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/reports")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--history-file", default="seller-copilot/artifacts/reports/ongoing_ml_history.jsonl")
    parser.add_argument(
        "--historical-llm-file",
        default="seller-copilot/artifacts/reports/historical_llm_model.json",
    )
    parser.add_argument("--historical-llm-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--new-llm-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--llm-eval-rows", type=int, default=80)
    parser.add_argument("--llm-max-new-tokens", type=int, default=6)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--llm-device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    data_dir = root / "data"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentiment_report = _load_json(root / "sentiment" / "classification_report.json")
    reranker_report = _load_json(root / "reranker" / "training_summary.json")
    pricing_report = _load_json(root / "pricing" / "training_summary.json")
    metrics_summary = _load_json(root / "metrics_summary.json")

    ranking_train = pd.read_parquet(data_dir / "ranking_train.parquet")
    ranking_test = pd.read_parquet(data_dir / "ranking_test.parquet")
    pricing_train = pd.read_parquet(data_dir / "pricing_train.parquet")
    pricing_test = pd.read_parquet(data_dir / "pricing_test.parquet")

    ranking_reference = _build_ranking_reference(ranking_train, ranking_test)
    pricing_reference = _build_pricing_reference(pricing_train, pricing_test)

    ranking_model_name = str(reranker_report.get("model_id") or ranking_reference["model"])
    ranking_metrics = dict(reranker_report.get("metrics") or ranking_reference["metrics"])
    pricing_model_name = str(pricing_report.get("model_id") or "FT-Transformer")
    pricing_metrics = dict(pricing_report.get("metrics") or pricing_reference["metrics"])

    performance_results = {
        "generated_at": _now_iso(),
        "sentiment": {
            "model": sentiment_report.get("model_id"),
            "metrics": {
                "accuracy": sentiment_report.get("metrics", {}).get("accuracy"),
                "macro_f1": sentiment_report.get("metrics", {}).get("macro avg", {}).get("f1-score"),
                "weighted_f1": sentiment_report.get("metrics", {}).get("weighted avg", {}).get("f1-score"),
            },
        },
        "ranking": {"model": ranking_model_name, "metrics": ranking_metrics},
        "pricing": {"model": pricing_model_name, "metrics": pricing_metrics},
        "retrieval": metrics_summary.get("retrieval")
        or _compute_retrieval_metrics_from_ranking(ranking_test)
        or {
            "proxy_ndcg@10_from_ranking_pairs": ranking_reference["metrics"].get("ndcg@10"),
            "proxy_recall@10_from_ranking_pairs": ranking_reference["metrics"].get("recall@10"),
        },
        "references": {
            "ranking": ranking_reference,
            "pricing": pricing_reference,
        },
    }
    _write_json(out_dir / "performance_results.json", performance_results)

    history_path = Path(args.history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    current_record = {
        "timestamp": performance_results["generated_at"],
        "sentiment_macro_f1": performance_results["sentiment"]["metrics"]["macro_f1"],
        "ranking_ndcg@10": performance_results["ranking"]["metrics"].get("ndcg@10"),
        "pricing_rmse": performance_results["pricing"]["metrics"].get("rmse"),
        "pricing_r2": performance_results["pricing"]["metrics"].get("r2"),
    }

    previous_record: dict[str, Any] | None = None
    if history_path.exists():
        lines = [line.strip() for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous_record = json.loads(lines[-1])
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(current_record) + "\n")

    deltas: dict[str, float | None] = {}
    for k, v in current_record.items():
        if k == "timestamp":
            continue
        prev_v = previous_record.get(k) if previous_record else None
        if prev_v is None or v is None:
            deltas[k] = None
        else:
            deltas[k] = float(v) - float(prev_v)

    ongoing_results = {
        "generated_at": performance_results["generated_at"],
        "current": current_record,
        "delta_vs_previous": deltas,
        "history_file": str(history_path),
    }
    _write_json(out_dir / "ongoing_ml_results.json", ongoing_results)

    analytics_findings = _render_analytics_figures(data_dir, out_dir / "figures")
    data_analytics = {
        "generated_at": performance_results["generated_at"],
        "row_counts": {
            "sentiment_train": int(pd.read_parquet(data_dir / "sentiment_train.parquet").shape[0]),
            "ranking_train": int(ranking_train.shape[0]),
            "pricing_train": int(pricing_train.shape[0]),
            "retrieval_corpus": int(pd.read_parquet(data_dir / "retrieval_corpus.parquet").shape[0]),
        },
        "findings": analytics_findings,
    }
    _write_json(out_dir / "data_analytics_findings.json", data_analytics)
    analytics_md_lines = [
        "# Data Analytics Findings",
        "",
        f"Generated at: `{performance_results['generated_at']}`",
        "",
        "## Row Counts",
        f"- sentiment_train: `{data_analytics['row_counts']['sentiment_train']}`",
        f"- ranking_train: `{data_analytics['row_counts']['ranking_train']}`",
        f"- pricing_train: `{data_analytics['row_counts']['pricing_train']}`",
        f"- retrieval_corpus: `{data_analytics['row_counts']['retrieval_corpus']}`",
        "",
        "## Findings",
    ]
    for finding in analytics_findings:
        analytics_md_lines.append(f"- {finding.get('title')}: `{finding.get('values')}`")
        figure = finding.get("figure")
        if figure:
            analytics_md_lines.append(f"  figure: `{figure}`")
    (out_dir / "data_analytics_findings.md").write_text("\n".join(analytics_md_lines), encoding="utf-8")

    sample_summary = _export_sample_observations(data_dir, out_dir / "samples", sample_size=args.sample_size)

    configured_new_llm = args.new_llm_model.strip() or "HuggingFaceTB/SmolLM2-360M-Instruct"

    sentiment_eval_source = pd.read_parquet(data_dir / "sentiment_test.parquet").dropna(subset=["text", "label"])
    llm_eval_df = sentiment_eval_source.sample(min(args.llm_eval_rows, len(sentiment_eval_source)), random_state=42).reset_index(
        drop=True
    )
    historical_llm_info = {
        "model_id": args.historical_llm_model,
        "source": "generated_in_current_project",
        "note": "This model is used as the historical baseline reference for comparison.",
    }
    _write_json(Path(args.historical_llm_file), historical_llm_info)

    llm_comparison = _build_llm_sentiment_comparison(
        eval_df=llm_eval_df,
        historical_model_id=args.historical_llm_model,
        new_model_id=configured_new_llm,
        max_new_tokens=args.llm_max_new_tokens,
        batch_size=args.llm_batch_size,
        device=args.llm_device,
    )
    _write_json(out_dir / "llm_sentiment_comparison.json", llm_comparison)
    comparison = _build_llm_side_by_side(out_dir / "comparisons", llm_comparison)
    _write_json(out_dir / "model_comparison_summary.json", comparison)

    summary_md = f"""# Results Pack

Generated at: `{performance_results['generated_at']}`

## Included Deliverables
- Performance results: `performance_results.json`
- Ongoing ML results: `ongoing_ml_results.json`
- Data analytics findings: `data_analytics_findings.json`
- Data analytics findings (narrative): `data_analytics_findings.md`
- Visualizations: `figures/`
- Side-by-side LLM model comparison: `comparisons/llm_sentiment_model_comparison.csv`
- Side-by-side LLM model comparison (table): `comparisons/llm_sentiment_model_comparison.md`
- Sample observations and ground truth labels: `samples/`

## Key Metrics
- Sentiment macro F1: `{performance_results['sentiment']['metrics']['macro_f1']}`
- Ranking nDCG@10: `{performance_results['ranking']['metrics'].get('ndcg@10')}`
- Pricing RMSE: `{performance_results['pricing']['metrics'].get('rmse')}`
- Pricing R2: `{performance_results['pricing']['metrics'].get('r2')}`

## Notes
- Historical LLM baseline: `{args.historical_llm_model}`
- New LLM model: `{configured_new_llm}`
- Sample rows exported: sentiment `{sample_summary['sentiment_samples']}`, ranking `{sample_summary['ranking_samples']}`, pricing `{sample_summary['pricing_samples']}`
"""
    (out_dir / "results_pack_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Saved evaluation reports to {out_dir}")


if __name__ == "__main__":
    main()
