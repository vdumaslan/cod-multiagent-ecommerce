#!/usr/bin/env python3
"""
Build agent Parquet (pools from local Home & Kitchen JSONL) + ops tables in one run.

From repo root:
  python seller-copilot/scripts/generate_data.py --products 8000
  python seller-copilot/scripts/generate_data.py --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_agent_dataset.build import (  # noqa: E402
    build_agent_tables_from_pools,
    load_pools_from_raw,
    write_agent_bundle,
)
import numpy as np  # noqa: E402
from synthetic_store.generator import SyntheticStoreConfig, generate_all  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate agent + ops Parquet data.")
    parser.add_argument("--products", type=int, default=8000, help="Number of distinct ASINs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--review-jsonl",
        default="seller-copilot/data/raw/amazon_reviews_2023/Home_and_Kitchen.jsonl",
    )
    parser.add_argument(
        "--meta-jsonl",
        default="seller-copilot/data/raw/amazon_reviews_2023/meta_Home_and_Kitchen.jsonl",
    )
    parser.add_argument(
        "--agent-dir",
        default="seller-copilot/data/agent_dataset",
    )
    parser.add_argument(
        "--synthetic-dir",
        default="seller-copilot/data/synthetic",
    )
    parser.add_argument("--pool-reviews", type=int, default=200_000)
    parser.add_argument("--pool-meta", type=int, default=120_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast preset: 1500 products, 25k/15k pools (minutes).",
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.quick or args.submission:
        args.products = 1500
        args.pool_reviews = 25_000
        args.pool_meta = 15_000

    cwd = Path.cwd()
    rev_path = (cwd / args.review_jsonl).resolve() if not Path(args.review_jsonl).is_absolute() else Path(args.review_jsonl)
    meta_path = (cwd / args.meta_jsonl).resolve() if not Path(args.meta_jsonl).is_absolute() else Path(args.meta_jsonl)

    pools = load_pools_from_raw(
        rev_path if rev_path.is_file() else None,
        meta_path if meta_path.is_file() else None,
        max_review_lines=args.pool_reviews,
        max_meta_lines=args.pool_meta,
    )

    rng = np.random.default_rng(args.seed)

    prods, revs, sig, retr, report = build_agent_tables_from_pools(
        pools,
        n_products=args.products,
        rng=rng,
        min_reviews_per_product=18,
        max_reviews_per_product=100,
    )

    manifest_extra = {
        "ingestion": {
            "source": "McAuley-Lab/Amazon-Reviews-2023 Home_and_Kitchen (sampled pools)",
            "review_pool_path": str(rev_path) if rev_path.is_file() else None,
            "meta_pool_path": str(meta_path) if meta_path.is_file() else None,
        },
        "quality_gates": {
            "min_reviews_per_product": 18,
            "replication_note": "Agent tables assembled from extract-style pools for downstream agents.",
        },
    }
    write_agent_bundle(
        prods,
        revs,
        sig,
        retr,
        agent_dir=Path(args.agent_dir),
        quality_report=report,
        manifest_extra=manifest_extra,
    )

    syn_cfg = SyntheticStoreConfig(random_seed=args.seed + 7, sales_history_days=180, n_suppliers=24)
    syn_out = generate_all(Path(args.agent_dir) / "products.parquet", Path(args.synthetic_dir), syn_cfg)
    print(json.dumps({"agent": report, "synthetic_row_counts": syn_out["row_counts"]}, indent=2))
    print("Done. Agent:", args.agent_dir, "Ops:", args.synthetic_dir)


if __name__ == "__main__":
    main()
