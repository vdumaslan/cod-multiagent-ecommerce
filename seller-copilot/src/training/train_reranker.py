from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader


def _compute_ranking_metrics(df: pd.DataFrame, score_col: str = "score", k: int = 10) -> dict[str, float]:
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []

    for _, group in df.groupby("query_text"):
        g = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        y_true = g["relevance_label"].astype(int).to_numpy()
        y_score = g[score_col].astype(float).to_numpy()
        if y_true.sum() == 0:
            continue

        ndcgs.append(float(ndcg_score([y_true], [y_score], k=min(k, len(y_true)))))

        topk = g.head(k)
        top_labels = topk["relevance_label"].astype(int).to_numpy()
        precisions.append(float(top_labels.mean()))
        recalls.append(float(1.0 if top_labels.sum() > 0 else 0.0))

        pos_idx = np.where(top_labels == 1)[0]
        mrrs.append(float(1.0 / (pos_idx[0] + 1)) if len(pos_idx) else 0.0)

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", default="seller-copilot/artifacts/data/ranking_train.parquet")
    parser.add_argument("--test-parquet", default="seller-copilot/artifacts/data/ranking_test.parquet")
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max-train-pairs", type=int, default=20000)
    parser.add_argument("--max-eval-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/reranker")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Ranking train/test parquet is empty.")

    if args.max_train_pairs and len(train_df) > args.max_train_pairs:
        train_df = train_df.sample(args.max_train_pairs, random_state=42).reset_index(drop=True)
    if args.max_eval_pairs and len(test_df) > args.max_eval_pairs:
        test_df = test_df.sample(args.max_eval_pairs, random_state=42).reset_index(drop=True)

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = CrossEncoder(args.model_id, num_labels=1, device=device)

    if args.epochs > 0:
        train_examples = [
            InputExample(texts=[q, d], label=float(y))
            for q, d, y in zip(
                train_df["query_text"].astype(str),
                train_df["product_document"].astype(str),
                train_df["relevance_label"].astype(float),
            )
        ]
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        model.fit(train_dataloader=train_loader, epochs=args.epochs, warmup_steps=100, show_progress_bar=True)
        model.save(str(out_dir / "model"))

    scores = model.predict(
        list(zip(test_df["query_text"].astype(str).tolist(), test_df["product_document"].astype(str).tolist())),
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    test_df = test_df.copy()
    test_df["score"] = np.array(scores, dtype=float)
    test_df["pred_label"] = (test_df["score"] >= 0.0).astype(int)

    metrics = _compute_ranking_metrics(test_df, score_col="score", k=args.k)
    pairwise_accuracy = float((test_df["pred_label"] == test_df["relevance_label"].astype(int)).mean())

    payload = {
        "model_id": args.model_id,
        "device": device,
        "num_train_pairs": int(len(train_df)),
        "num_eval_pairs": int(len(test_df)),
        "epochs": int(args.epochs),
        "metrics": {**metrics, "pairwise_accuracy": pairwise_accuracy},
    }
    (out_dir / "training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
