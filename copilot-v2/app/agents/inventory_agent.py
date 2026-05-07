"""Inventory / stock classification from precomputed cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.bigquery_cache import (
    fallback_to_local_enabled,
    load_inventory_cache,
    selected_data_backend,
)

SNAPSHOT_ID = "38710839ca6e1009"


class InventoryAgent:
    def __init__(
        self,
        snapshot_id: str = SNAPSHOT_ID,
        artifacts_root: Path | None = None,
        data_backend: str | None = None,
    ) -> None:
        self._cache_source = "local"
        backend = selected_data_backend(data_backend)
        if backend == "bigquery":
            try:
                self._cache = load_inventory_cache(snapshot_id)
                self._cache_source = "bigquery"
                return
            except Exception:
                if not fallback_to_local_enabled():
                    raise
                self._cache_source = "local_fallback"

        root = artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
        cache_path = root / "caches" / snapshot_id / "inventory" / "inventory_cache.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(f"Inventory cache not found at {cache_path}. Run precompute_inventory.py first.")
        cols = ["product_id", "stock_status", "risk_flag", "on_hand_units",
                "safety_stock_units", "available_to_sell", "mean_daily_revenue", "total_returns"]
        df = pd.read_parquet(cache_path, columns=cols)
        self._cache: dict[str, dict[str, object]] = {
            str(row["product_id"]): {
                "stock_status": row["stock_status"],
                "risk_flag": bool(row["risk_flag"]),
                "on_hand_units": float(row["on_hand_units"]),
                "safety_stock_units": float(row["safety_stock_units"]),
                "available_to_sell": float(row["available_to_sell"]),
                "mean_daily_revenue": float(row["mean_daily_revenue"]),
                "total_returns": float(row["total_returns"]),
            }
            for _, row in df.iterrows()
        }

    @property
    def cache_source(self) -> str:
        return self._cache_source

    def lookup(self, product_id: str) -> dict[str, object]:
        pid = str(product_id)
        if pid not in self._cache:
            return {"product_id": pid, "found": False, "stock_status": "unknown", "risk_flag": False}
        return {"product_id": pid, "found": True, **self._cache[pid]}

