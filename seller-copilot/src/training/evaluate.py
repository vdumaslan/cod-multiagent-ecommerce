from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_metrics_from_ranking(
    ranking_df: pd.DataFrame,
    embed_model_id: str,
    k: int = 10,
    max_rows: int = 30000,
) -> dict[str, float]:
    df = ranking_df.copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    model = SentenceTransformer(embed_model_id)
    unique_docs = df["product_document"].astype(str).drop_duplicates().tolist()
    unique_queries = df["query_text"].astype(str).drop_duplicates().tolist()
    doc_emb = model.encode(unique_docs, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    qry_emb = model.encode(unique_queries, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

    doc_index = {d: i for i, d in enumerate(unique_docs)}
    qry_index = {q: i for i, q in enumerate(unique_queries)}

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []

    for query, group in df.groupby("query_text"):
        labels = group["relevance_label"].astype(int).to_numpy()
        if labels.sum() == 0:
            continue
        docs = group["product_document"].astype(str).tolist()
        qv = qry_emb[qry_index[str(query)]]
        dv = np.vstack([doc_emb[doc_index[d]] for d in docs])
        scores = dv @ qv
        order = np.argsort(-scores)
        top = labels[order][:k]

        recalls.append(float(1.0 if top.sum() > 0 else 0.0))
        hit_rates.append(float(1.0 if top.sum() > 0 else 0.0))
        pos_idx = np.where(top == 1)[0]
        mrrs.append(float(1.0 / (pos_idx[0] + 1)) if len(pos_idx) else 0.0)
        ndcgs.append(float(ndcg_score([labels], [scores], k=min(k, len(labels)))))

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment-report", default="seller-copilot/artifacts/sentiment/classification_report.json")
    parser.add_argument("--reranker-report", default="seller-copilot/artifacts/reranker/training_summary.json")
    parser.add_argument("--pricing-report", default="seller-copilot/artifacts/pricing/training_summary.json")
    parser.add_argument("--ranking-test-parquet", default="seller-copilot/artifacts/data/ranking_test.parquet")
    parser.add_argument("--embed-model-id", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--retrieval-max-rows", type=int, default=30000)
    parser.add_argument("--output", default="seller-copilot/artifacts/metrics_summary.json")
    args = parser.parse_args()

    sentiment = _safe_load_json(Path(args.sentiment_report))
    reranker = _safe_load_json(Path(args.reranker_report))
    pricing = _safe_load_json(Path(args.pricing_report))

    ranking_test = pd.read_parquet(args.ranking_test_parquet)
    retrieval_metrics = _retrieval_metrics_from_ranking(
        ranking_df=ranking_test,
        embed_model_id=args.embed_model_id,
        k=args.k,
        max_rows=args.retrieval_max_rows,
    )

    metrics = {
        "retrieval": retrieval_metrics,
        "ranking": reranker.get("metrics", {}),
        "sentiment": {
            "macro_f1": sentiment.get("metrics", {}).get("macro avg", {}).get("f1-score"),
            "accuracy": sentiment.get("metrics", {}).get("accuracy"),
        },
        "pricing": pricing.get("metrics", {}),
        "end_to_end": {
            "model_artifacts_ready": all(
                [
                    bool(sentiment),
                    bool(reranker),
                    bool(pricing),
                ]
            )
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
