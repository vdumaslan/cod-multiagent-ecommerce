#!/usr/bin/env python3
"""
Build ops-only Parquet (inventory, sales, suppliers, …) joined to products.parquet.

Run ``generate_data.py`` first, or Stage 1, so ``data/agent_dataset/products.parquet`` exists.

Example:
  python seller-copilot/scripts/generate_ops_data.py
  python seller-copilot/scripts/generate_ops_data.py --seed 7 --days 365
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_store.generator import SyntheticStoreConfig, generate_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ops Parquet aligned to agent products.")
    parser.add_argument(
        "--products",
        default="seller-copilot/data/agent_dataset/products.parquet",
        help="Path to curated products.parquet",
    )
    parser.add_argument(
        "--out",
        default="seller-copilot/data/synthetic",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=180, help="Sales + marketing history length")
    parser.add_argument("--suppliers", type=int, default=24)
    args = parser.parse_args()

    products_path = Path(args.products)
    if not products_path.is_file():
        print(
            f"Missing {products_path}. Run generate_data.py or Stage 1 first:\n"
            "  python seller-copilot/scripts/generate_data.py --quick",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = SyntheticStoreConfig(
        random_seed=args.seed,
        sales_history_days=args.days,
        n_suppliers=args.suppliers,
    )
    manifest = generate_all(products_path, Path(args.out), cfg)
    print(json.dumps(manifest["row_counts"], indent=2))
    print(f"Wrote manifest: {Path(args.out) / 'synthetic_store_manifest.json'}")


if __name__ == "__main__":
    main()
