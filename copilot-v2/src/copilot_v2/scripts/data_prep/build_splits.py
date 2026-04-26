from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _assign_split(product_ids: np.ndarray, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = rng.random(len(product_ids))
    split = np.where(u < train_frac, "train", np.where(u < train_frac + val_frac, "val", "test"))
    return pd.DataFrame({"product_id": product_ids, "split": split})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.train_frac <= 0 or args.val_frac <= 0 or (args.train_frac + args.val_frac) >= 1.0:
        raise SystemExit("Invalid split fractions: require train>0, val>0, train+val<1")

    repo_root = Path(__file__).resolve().parents[4]
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root
    snap_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)
    out_dir = artifacts_root / "splits" / str(args.snapshot_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    products_path = snap_dir / "products.parquet"
    reviews_path = snap_dir / "reviews.parquet"
    if not products_path.exists() or not reviews_path.exists():
        raise SystemExit(f"Missing products/reviews parquet under {snap_dir}")

    products = pd.read_parquet(products_path, columns=["product_id"])
    reviews = pd.read_parquet(reviews_path, columns=["product_id"])

    products["product_id"] = products["product_id"].astype(str)
    reviews["product_id"] = reviews["product_id"].astype(str)

    split_df = _assign_split(products["product_id"].to_numpy(), args.seed, args.train_frac, args.val_frac)

    products_split = products.merge(split_df, on="product_id", how="left")
    reviews_split = reviews.merge(split_df, on="product_id", how="left")

    products_out = out_dir / "products_split.parquet"
    reviews_out = out_dir / "reviews_split.parquet"
    products_split.to_parquet(products_out, index=False)
    reviews_split.to_parquet(reviews_out, index=False)

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "strategy": "random_by_product_id",
        "fractions": {"train": float(args.train_frac), "val": float(args.val_frac), "test": float(1.0 - args.train_frac - args.val_frac)},
        "seed": int(args.seed),
        "counts": {
            "products": products_split["split"].value_counts().to_dict(),
            "reviews": reviews_split["split"].value_counts().to_dict(),
        },
    }
    (out_dir / "split_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

