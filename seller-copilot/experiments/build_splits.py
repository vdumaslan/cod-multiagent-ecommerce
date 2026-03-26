#!/usr/bin/env python3
"""
Build reproducible train/val/test splits.

- Reviews: time-ordered quantile cut on event_ts (70% / 15% / 15%).
- Products: time-ordered quantile cut on last_review_ts (same ratios).

If timestamps were unusable, falls back to deterministic hash split on review_id (not expected for current data).

Writes:
  artifacts/splits/split_config.json
  artifacts/splits/reviews_split.parquet
  artifacts/splits/products_split.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import COPILOT_ROOT, DATA_AGENT, SPLITS_DIR


def _hash_split_ids(ids: pd.Series, train: float, val: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    h = ids.astype(str).map(lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16) % (2**31))
    u = (h % 10000) / 10000.0
    out = np.where(u < train, "train", np.where(u < train + val, "val", "test"))
    return pd.Series(out, index=ids.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--agent-dir", type=Path, default=DATA_AGENT)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--force-hash-reviews", action="store_true", help="Debug: hash split for reviews")
    args = parser.parse_args()
    if args.train + args.val >= 1.0:
        print("train + val must be < 1", file=sys.stderr)
        sys.exit(1)

    agent = args.agent_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rev = pd.read_parquet(agent / "reviews.parquet")
    prod = pd.read_parquet(agent / "products.parquet")

    test = 1.0 - args.train - args.val

    if args.force_hash_reviews or rev["event_ts"].isna().all():
        rev = rev.copy()
        rev["split"] = _hash_split_ids(rev["review_id"], args.train, args.val, args.seed)
        split_mode = "hash_review_id"
        t0 = t1 = None
    else:
        rev = rev.sort_values("event_ts").reset_index(drop=True)
        ts = rev["event_ts"].astype("int64").to_numpy()
        q1 = float(np.quantile(ts, args.train))
        q2 = float(np.quantile(ts, args.train + args.val))
        rev["split"] = np.where(
            ts <= q1, "train", np.where(ts <= q2, "val", "test")
        )
        split_mode = "time_event_ts"
        t0 = pd.Timestamp(rev["event_ts"].min())
        t1 = pd.Timestamp(rev["event_ts"].max())

    prod = prod.sort_values("last_review_ts").reset_index(drop=True)
    pts = prod["last_review_ts"].astype("int64").to_numpy()
    pq1 = float(np.quantile(pts, args.train))
    pq2 = float(np.quantile(pts, args.train + args.val))
    prod["split"] = np.where(pts <= pq1, "train", np.where(pts <= pq2, "val", "test"))

    cfg = {
        "random_seed": args.seed,
        "train_frac": args.train,
        "val_frac": args.val,
        "test_frac": test,
        "review_split_mode": split_mode,
        "review_time_range": [str(t0) if t0 is not None else None, str(t1) if t1 is not None else None],
        "paths": {
            "reviews_split": str(out / "reviews_split.parquet"),
            "products_split": str(out / "products_split.parquet"),
        },
    }
    (out / "split_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    rev[["review_id", "product_id", "split"]].to_parquet(out / "reviews_split.parquet", index=False)
    prod[["product_id", "split"]].to_parquet(out / "products_split.parquet", index=False)

    print(json.dumps({"split_config": cfg, "review_counts": rev["split"].value_counts().to_dict()}, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    # Allow running as script from experiments/
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
