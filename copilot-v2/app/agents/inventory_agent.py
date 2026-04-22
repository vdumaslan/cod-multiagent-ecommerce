"""Inventory / stock classification from precomputed cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"


class InventoryAgent:
    def __init__(self, snapshot_id: str = SNAPSHOT_ID, artifacts_root: Path | None = None) -> None:
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

    def lookup(self, product_id: str) -> dict[str, object]:
        pid = str(product_id)
        if pid not in self._cache:
            return {"product_id": pid, "found": False, "stock_status": "unknown", "risk_flag": False}
        return {"product_id": pid, "found": True, **self._cache[pid]}

    def lookup_many(self, product_ids: list[str]) -> list[dict[str, object]]:
        return [self.lookup(pid) for pid in product_ids]
