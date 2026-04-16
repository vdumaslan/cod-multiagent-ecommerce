from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--price-col", default="", help="Optional explicit price column (else auto-detect price_usd then price).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root
    snap_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)

    products_path = snap_dir / "products.parquet"
    reviews_path = snap_dir / "reviews.parquet"
    signals_path = snap_dir / "product_signals.parquet"
    retrieval_path = snap_dir / "retrieval_corpus.parquet"
    if not products_path.exists():
        raise SystemExit(f"Missing {products_path}")

    products = pd.read_parquet(products_path)
    price_col = args.price_col.strip()
    if not price_col:
        price_col = "price_usd" if "price_usd" in products.columns else ("price" if "price" in products.columns else "")
    if not price_col or price_col not in products.columns:
        raise SystemExit(f"Could not find a price column in products.parquet (tried {price_col!r}).")

    products[price_col] = pd.to_numeric(products[price_col], errors="coerce")
    priced = products.loc[products[price_col].notna()].copy()

    out_dir = snap_dir / "priced_subset"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_ids = set(priced["product_id"].astype(str).tolist())

    priced.to_parquet(out_dir / "products.parquet", index=False)

    if reviews_path.exists():
        reviews = pd.read_parquet(reviews_path)
        reviews = reviews.loc[reviews["product_id"].astype(str).isin(keep_ids)].copy()
        reviews.to_parquet(out_dir / "reviews.parquet", index=False)

    if signals_path.exists():
        signals = pd.read_parquet(signals_path)
        signals = signals.loc[signals["product_id"].astype(str).isin(keep_ids)].copy()
        signals.to_parquet(out_dir / "product_signals.parquet", index=False)

    if retrieval_path.exists():
        retrieval = pd.read_parquet(retrieval_path)
        retrieval = retrieval.loc[retrieval["product_id"].astype(str).isin(keep_ids)].copy()
        retrieval.to_parquet(out_dir / "retrieval_corpus.parquet", index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "price_col": price_col,
        "rows": {"products": int(len(priced))},
        "output_dir": str(out_dir),
    }
    (out_dir / "manifest_priced_subset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "products": int(len(priced))}, indent=2))


if __name__ == "__main__":
    main()

