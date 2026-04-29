"""Pricing recommendations from precomputed TabPFN cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"


class PricingAgent:
    def __init__(self, snapshot_id: str = SNAPSHOT_ID, artifacts_root: Path | None = None) -> None:
        root = artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
        cache_path = root / "caches" / snapshot_id / "pricing" / "pricing_cache.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(f"Pricing cache not found at {cache_path}. Run precompute_pricing.py first.")
        df = pd.read_parquet(cache_path, columns=["product_id", "predicted_price_change_pct"])
        self._cache: dict[str, float] = dict(
            zip(df["product_id"].astype(str), df["predicted_price_change_pct"].astype(float))
        )

    def lookup(self, product_id: str) -> dict[str, object]:
        pid = str(product_id)
        if pid not in self._cache:
            return {"product_id": pid, "predicted_price_change_pct": None, "found": False}
        return {"product_id": pid, "predicted_price_change_pct": self._cache[pid], "found": True}

