"""Synthetic operational store data aligned to curated Amazon product_ids (COGS, inventory, sales, suppliers)."""

from synthetic_store.generator import SyntheticStoreConfig, generate_all

__all__ = ["SyntheticStoreConfig", "generate_all"]
