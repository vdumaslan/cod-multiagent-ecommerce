"""Sentiment signals from precomputed DistilRoBERTa cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"


class SentimentAgent:
    def __init__(self, snapshot_id: str = SNAPSHOT_ID, artifacts_root: Path | None = None) -> None:
        root = artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
        cache_path = root / "caches" / snapshot_id / "sentiment" / "sentiment_cache.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(f"Sentiment cache not found at {cache_path}. Run precompute_sentiment.py first.")
        df = pd.read_parquet(cache_path, columns=["product_id", "n_reviews", "p_pos", "p_neu", "p_neg"])
        self._cache: dict[str, dict[str, float]] = {
            str(row["product_id"]): {
                "n_reviews": float(row["n_reviews"]),
                "p_pos": float(row["p_pos"]),
                "p_neu": float(row["p_neu"]),
                "p_neg": float(row["p_neg"]),
            }
            for _, row in df.iterrows()
        }

    def lookup(self, product_id: str) -> dict[str, object]:
        pid = str(product_id)
        if pid not in self._cache:
            return {"product_id": pid, "found": False, "n_reviews": None, "p_pos": None, "p_neu": None, "p_neg": None}
        return {"product_id": pid, "found": True, **self._cache[pid]}

    def lookup_many(self, product_ids: list[str]) -> list[dict[str, object]]:
        return [self.lookup(pid) for pid in product_ids]
