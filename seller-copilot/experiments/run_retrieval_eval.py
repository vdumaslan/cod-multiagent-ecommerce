#!/usr/bin/env python3
"""
Retrieval: two sentence-transformers + FAISS (inner product on L2-normalized vectors = cosine).

Query = product title; corpus = product_document. Self-match masked out when scoring.

Metrics: Recall@K, nDCG@K, MRR for test products.

Writes artifacts/evals/retrieval/ and artifacts/faiss/ (default model index for Streamlit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from _paths import DATA_AGENT, EVALS_DIR, FAISS_DIR, SPLITS_DIR


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / (ranks + 1)))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """Binary relevance: only rank 0 is relevant."""
    dcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
    # ideal DCG = 1
    return float(np.mean(dcg / 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs=2,
        default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = DATA_AGENT
    prod = pd.read_parquet(agent / "products.parquet")
    corp = pd.read_parquet(agent / "retrieval_corpus.parquet")
    psplit = pd.read_parquet(SPLITS_DIR / "products_split.parquet")
    prod = prod.merge(psplit, on="product_id", how="inner")
    corp = corp.merge(prod[["product_id", "title", "split"]], on="product_id", how="inner")

    # Align corpus rows to fixed order
    order = corp["product_id"].tolist()
    id_to_idx = {pid: i for i, pid in enumerate(order)}
    docs = corp["product_document"].astype(str).tolist()
    test_mask = corp["split"] == "test"
    test_indices = np.where(test_mask.to_numpy())[0]

    out_eval = EVALS_DIR / "retrieval"
    out_eval.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in args.models:
        mdl = SentenceTransformer(model_name)
        doc_emb = mdl.encode(docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = doc_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_emb.astype(np.float32))

        queries = corp["title"].astype(str).tolist()
        q_emb = mdl.encode(queries, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        ranks = []
        for qi in test_indices:
            # Query = title embedding; corpus = product_document — not duplicate vectors; do not mask.
            sims = (doc_emb @ q_emb[qi]).astype(np.float64)
            order_r = np.argsort(-sims)
            rank = int(np.where(order_r == qi)[0][0])
            ranks.append(rank)
        ranks = np.array(ranks, dtype=np.int64)

        k = args.k
        results[model_name] = {
            "recall_at_k": recall_at_k(ranks, k),
            "ndcg_at_k": ndcg_at_k(ranks, k),
            "mrr": mrr(ranks),
            "n_test": len(ranks),
            "k": k,
        }

        # Persist first model FAISS + metadata for app
        if model_name == args.models[0]:
            faiss.write_index(index, str(FAISS_DIR / "index_flatip.faiss"))
            meta = {
                "model_name": model_name,
                "dim": dim,
                "n_vectors": len(order),
                "corpus_parquet": "corpus.parquet",
            }
            (FAISS_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            pd.DataFrame({"product_id": order, "product_document": docs}).to_parquet(
                FAISS_DIR / "corpus.parquet", index=False
            )
            (FAISS_DIR / "model_name.txt").write_text(model_name, encoding="utf-8")

    (out_eval / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
