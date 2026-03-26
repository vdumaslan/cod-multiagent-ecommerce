from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def _norm01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _expand_query(query: str) -> str:
    q = query.lower()
    extras: list[str] = []
    if "revenue" in q or "sales" in q or "growth" in q:
        extras.extend(["demand", "high rating", "popular", "review"])
    if "margin" in q or "profit" in q:
        extras.extend(["value", "price", "quality", "cost"])
    if "inventory" in q or "stock" in q:
        extras.extend(["supply", "reorder", "availability"])
    if "cost" in q:
        extras.extend(["unit cost", "efficiency"])
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


def load_retriever(faiss_dir: Path) -> dict[str, Any]:
    model_name = (faiss_dir / "model_name.txt").read_text(encoding="utf-8").strip()
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(faiss_dir / "index_flatip.faiss"))
    corpus = pd.read_parquet(faiss_dir / "corpus.parquet")
    docs = corpus["product_document"].astype(str).tolist()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(docs)
    return {
        "model_name": model_name,
        "model": model,
        "index": index,
        "corpus": corpus,
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
    }


def retrieve_evidence(
    query: str,
    retriever: dict[str, Any],
    k: int = 8,
    *,
    candidate_k: int = 80,
    dense_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> pd.DataFrame:
    expanded = _expand_query(query)

    # Dense retrieval: high recall candidate set
    q_dense = retriever["model"].encode([expanded], normalize_embeddings=True)
    dense_scores, dense_idx = retriever["index"].search(q_dense.astype(np.float32), candidate_k)
    cand_idx = dense_idx[0]
    cand_dense = dense_scores[0].astype(np.float64)

    # Lexical rerank on same candidate pool
    q_lex = retriever["tfidf"].transform([expanded])
    lex_all = (retriever["tfidf_matrix"] @ q_lex.T).toarray().ravel()
    cand_lex = lex_all[cand_idx]

    dense_n = _norm01(cand_dense)
    lex_n = _norm01(cand_lex)
    hybrid = dense_weight * dense_n + lexical_weight * lex_n
    order = np.argsort(-hybrid)
    keep = order[:k]

    out = retriever["corpus"].iloc[cand_idx[keep]].copy().reset_index(drop=True)
    out["score"] = hybrid[keep]
    out["score_dense"] = cand_dense[keep]
    out["score_lexical"] = cand_lex[keep]
    out["query_expanded"] = expanded
    return out
