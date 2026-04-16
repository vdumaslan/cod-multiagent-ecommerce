from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--max-positive-multiplier", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root
    snap_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)

    reviews_path = snap_dir / "reviews.parquet"
    if not reviews_path.exists():
        raise SystemExit(f"Missing {reviews_path}")

    reviews = pd.read_parquet(reviews_path)
    if "rating" not in reviews.columns:
        raise SystemExit("reviews.parquet missing rating column")

    r = reviews.copy()
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce")

    # Buckets: neg <=2, neu==3, pos>=4
    bucket = np.where(r["rating"] <= 2, "neg", np.where(r["rating"] >= 4, "pos", "neu"))
    r["sent_bucket"] = bucket

    counts = r["sent_bucket"].value_counts(dropna=False).to_dict()
    neg = int(counts.get("neg", 0))
    neu = int(counts.get("neu", 0))
    pos = int(counts.get("pos", 0))

    target_pos = int(min(pos, args.max_positive_multiplier * max(neg, neu, 1)))

    rng = np.random.default_rng(args.seed)
    keep_idx: list[int] = []
    for b in ["neg", "neu"]:
        keep_idx.extend(r.index[r["sent_bucket"] == b].tolist())
    pos_idx = r.index[r["sent_bucket"] == "pos"].to_numpy()
    if len(pos_idx) > target_pos:
        pos_keep = rng.choice(pos_idx, size=target_pos, replace=False)
        keep_idx.extend(pos_keep.tolist())
    else:
        keep_idx.extend(pos_idx.tolist())

    balanced = r.loc[keep_idx].drop(columns=["sent_bucket"])
    balanced = balanced.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_path = snap_dir / "reviews_sentiment_balanced.parquet"
    balanced.to_parquet(out_path, index=False)

    meta = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "in_rows": int(len(reviews)),
        "out_rows": int(len(balanced)),
        "bucket_counts_in": {"neg": neg, "neu": neu, "pos": pos},
        "max_positive_multiplier": float(args.max_positive_multiplier),
        "target_pos": int(target_pos),
        "seed": int(args.seed),
        "output": str(out_path),
    }
    (snap_dir / "reviews_sentiment_balanced_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_path": str(out_path), "out_rows": int(len(balanced))}, indent=2))


if __name__ == "__main__":
    main()

