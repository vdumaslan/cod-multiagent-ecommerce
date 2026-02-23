from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class DiscoveryAgent:
    def __init__(self, embed_model_id: str, index_path: str, lookup_path: str) -> None:
        self.embedder = SentenceTransformer(embed_model_id)
        self.index = faiss.read_index(index_path)
        self.lookup = pd.read_parquet(lookup_path)

    def retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q = q.astype("float32")
        scores, idx = self.index.search(q, top_k)
        out = self.lookup.iloc[idx[0]].copy().reset_index(drop=True)
        out["retrieval_score"] = scores[0]
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="seller-copilot/artifacts/retrieval/retrieval_index.faiss")
    parser.add_argument("--lookup", default="seller-copilot/artifacts/retrieval/retrieval_lookup.parquet")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    agent = DiscoveryAgent(args.embed_model, args.index, args.lookup)
    results = agent.retrieve(args.query, top_k=10)
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


