#!/usr/bin/env python3
"""
Validate curated agent + synthetic datasets: files exist, row counts, join keys, basic stats.

Exit code 0 = all checks passed; non-zero = failure or warnings-as-errors with --strict.

Usage:
  python seller-copilot/scripts/validate_agent_data.py
  python seller-copilot/scripts/validate_agent_data.py --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COPILOT_ROOT = Path(__file__).resolve().parents[1]
AGENT = COPILOT_ROOT / "data" / "agent_dataset"
SYN = COPILOT_ROOT / "data" / "synthetic"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failure")
    parser.add_argument("--agent-dir", type=Path, default=AGENT)
    parser.add_argument("--synthetic-dir", type=Path, default=SYN)
    args = parser.parse_args()

    failed = False
    agent = args.agent_dir.resolve()
    syn = args.synthetic_dir.resolve()

    required_agent = [
        "products.parquet",
        "reviews.parquet",
        "product_signals.parquet",
        "retrieval_corpus.parquet",
        "agent_dataset_manifest.json",
    ]
    required_syn = [
        "inventory_skus.parquet",
        "suppliers.parquet",
        "product_supplier_map.parquet",
        "sales_daily.parquet",
        "marketing_spend_daily.parquet",
        "store_kpis_weekly.parquet",
        "synthetic_store_manifest.json",
    ]

    for name in required_agent:
        p = agent / name
        if not p.is_file():
            _fail(f"missing agent file: {p}")
            failed = True
    for name in required_syn:
        p = syn / name
        if not p.is_file():
            _fail(f"missing synthetic file: {p}")
            failed = True

    if failed:
        print("\nValidation summary: FAILED (missing files)")
        sys.exit(1)

    # Load and validate joins
    if (agent / "products.parquet").is_file():
        prod = pd.read_parquet(agent / "products.parquet")
        _ok(f"products: {len(prod):,} rows, columns={list(prod.columns)}")
        if len(prod) < 100:
            _warn("very few products - run generate_data.py with higher --products if not a smoke test.")
        if "product_id" not in prod.columns:
            _fail("products missing product_id")
            failed = True

    if (agent / "reviews.parquet").is_file():
        rev = pd.read_parquet(agent / "reviews.parquet")
        _ok(f"reviews: {len(rev):,} rows")
        if "product_id" not in rev.columns:
            _fail("reviews missing product_id")
            failed = True

    if (agent / "products.parquet").is_file() and (agent / "reviews.parquet").is_file():
        prod = pd.read_parquet(agent / "products.parquet")
        rev = pd.read_parquet(agent / "reviews.parquet")
        ps = set(prod["product_id"].astype(str))
        rs = set(rev["product_id"].astype(str))
        orphan = rs - ps
        if orphan:
            _warn(f"{len(orphan)} review product_ids not in products (unexpected)")
            if args.strict:
                _fail("review product_ids not subset of products")
                failed = True
        else:
            _ok("reviews product_id is subset of products")

    if (agent / "agent_dataset_manifest.json").is_file():
        m = json.loads((agent / "agent_dataset_manifest.json").read_text(encoding="utf-8"))
        _ok(f"agent manifest: row_counts={m.get('row_counts', m)}")

    if (syn / "inventory_skus.parquet").is_file() and (agent / "products.parquet").is_file():
        inv = pd.read_parquet(syn / "inventory_skus.parquet")
        prod = pd.read_parquet(agent / "products.parquet")
        _ok(f"inventory_skus: {len(inv):,} rows")
        if set(inv["product_id"].astype(str)) != set(prod["product_id"].astype(str)):
            _warn("inventory_skus product_ids may not match products (re-run synthetic after Stage 1)")
            if args.strict:
                _fail("synthetic inventory product_id mismatch")
                failed = True
        else:
            _ok("synthetic inventory and products: same product_id set")

    if (syn / "sales_daily.parquet").is_file():
        s = pd.read_parquet(syn / "sales_daily.parquet")
        _ok(f"sales_daily: {len(s):,} rows")

    if (agent / "products.parquet").is_file():
        prod = pd.read_parquet(agent / "products.parquet")
        if "price" in prod.columns:
            p = pd.to_numeric(prod["price"], errors="coerce")
            valid = p.dropna()
            if len(valid):
                _ok(
                    f"price stats: min={valid.min():.2f} max={valid.max():.2f} "
                    f"median={valid.median():.2f} null_pct={100 * p.isna().mean():.1f}%"
                )
                vc = valid.round(2).value_counts().head(3)
                _ok(f"top price values (rounded): {vc.to_dict()}")

    print("\nValidation summary: " + ("FAILED" if failed else "PASSED"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
