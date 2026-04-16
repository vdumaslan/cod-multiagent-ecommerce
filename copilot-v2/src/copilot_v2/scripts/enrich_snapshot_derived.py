from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        return v
    return None


_PACK_RE = re.compile(
    r"(?P<n>\d{1,3})\s*(?:pack|pk|count|ct|pcs|pieces|pc)\b",
    flags=re.IGNORECASE,
)


def _infer_pack_units(title: Any) -> float:
    if not isinstance(title, str):
        return float("nan")
    m = _PACK_RE.search(title)
    if not m:
        return float("nan")
    try:
        n = int(m.group("n"))
    except Exception:
        return float("nan")
    if n <= 0:
        return float("nan")
    return float(n)


def _build_product_document(row: pd.Series) -> str:
    title = str(_coalesce(row.get("title"), ""))[:500]
    brand = str(_coalesce(row.get("brand"), ""))[:200]
    category = str(_coalesce(row.get("category"), ""))[:200]
    subcategory = str(_coalesce(row.get("subcategory"), ""))[:200]

    price = _coalesce(row.get("price_usd"), row.get("price"))
    try:
        price_f = float(price) if price is not None else float("nan")
    except Exception:
        price_f = float("nan")

    avg_rating = _coalesce(row.get("avg_rating"), row.get("rating_mean"))
    try:
        avg_rating_f = float(avg_rating) if avg_rating is not None else float("nan")
    except Exception:
        avg_rating_f = float("nan")

    rating_count = _coalesce(row.get("rating_count"), row.get("n_ratings"))
    try:
        rating_count_i = int(rating_count) if rating_count is not None else None
    except Exception:
        rating_count_i = None

    desc = str(_coalesce(row.get("description"), row.get("product_description"), ""))[:15000]

    lines = [
        f"title: {title}",
        f"brand: {brand}",
        f"category: {category}",
        f"subcategory: {subcategory}",
        f"price_usd: {price_f:.2f}" if not math.isnan(price_f) else "price_usd: null",
        f"avg_rating: {avg_rating_f:.2f}" if not math.isnan(avg_rating_f) else "avg_rating: null",
        f"rating_count: {rating_count_i}" if rating_count_i is not None else "rating_count: null",
        f"description: {desc}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--write-inplace", action="store_true", help="Overwrite snapshot parquet files in place.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    artifacts_root = (repo_root / args.artifacts_root).resolve() if not args.artifacts_root.is_absolute() else args.artifacts_root
    snap_dir = artifacts_root / "data_snapshots" / str(args.snapshot_id)

    products_path = snap_dir / "products.parquet"
    signals_path = snap_dir / "product_signals.parquet"
    retrieval_path = snap_dir / "retrieval_corpus.parquet"

    if not products_path.exists() or not signals_path.exists():
        raise SystemExit(f"Missing products/signals parquet under {snap_dir}")

    products = pd.read_parquet(products_path)
    signals = pd.read_parquet(signals_path)

    if "product_id" not in products.columns:
        raise SystemExit("products.parquet missing product_id")
    if "product_id" not in signals.columns:
        raise SystemExit("product_signals.parquet missing product_id")

    # 1) Standardize / refresh product_document for retrieval clarity.
    products = products.copy()
    products["product_document"] = products.apply(_build_product_document, axis=1)

    # 2) Subcategory-relative enrichments on product_signals (joins on product_id).
    sig = signals.merge(
        products[["product_id", "subcategory", "title"] + ([c for c in ["price", "price_usd", "avg_rating", "rating_count"] if c in products.columns])],
        on="product_id",
        how="left",
        suffixes=("", "_p"),
    )

    # Choose numeric price column.
    if "price_usd" in sig.columns:
        price_col = "price_usd"
    elif "price" in sig.columns:
        price_col = "price"
    else:
        price_col = None

    if price_col is not None:
        sig[price_col] = pd.to_numeric(sig[price_col], errors="coerce")

    sig["inferred_pack_units"] = sig["title"].apply(_infer_pack_units)
    if price_col is not None:
        sig["price_per_unit"] = sig[price_col] / sig["inferred_pack_units"]
    else:
        sig["price_per_unit"] = np.nan

    def _add_group_stats(df: pd.DataFrame) -> pd.DataFrame:
        if price_col is None:
            df["subcategory_median_price"] = np.nan
            df["subcategory_mean_price"] = np.nan
            df["price_percentile_in_subcategory"] = np.nan
        else:
            df["subcategory_median_price"] = df[price_col].median(skipna=True)
            df["subcategory_mean_price"] = df[price_col].mean(skipna=True)
            try:
                # rank(pct=True) gives (0,1]; convert to [0,100]
                df["price_percentile_in_subcategory"] = df[price_col].rank(pct=True) * 100.0
            except Exception:
                df["price_percentile_in_subcategory"] = np.nan

        df["subcategory_product_count"] = len(df)
        df["subcategory_mean_price_per_unit"] = df["price_per_unit"].mean(skipna=True)

        # Ratings: prefer product_signals if present, else products
        rating_col = "avg_rating" if "avg_rating" in df.columns else None
        if rating_col:
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
            df["subcategory_mean_rating"] = df[rating_col].mean(skipna=True)
            df["rating_vs_subcategory_mean"] = df[rating_col] - df["subcategory_mean_rating"]
        else:
            df["subcategory_mean_rating"] = np.nan
            df["rating_vs_subcategory_mean"] = np.nan

        return df

    sig = sig.groupby("subcategory", dropna=False, group_keys=False).apply(_add_group_stats)

    # Only persist back to product_signals columns + new derived columns.
    derived_cols = [
        "price_percentile_in_subcategory",
        "rating_vs_subcategory_mean",
        "subcategory_median_price",
        "subcategory_mean_price",
        "subcategory_mean_rating",
        "subcategory_product_count",
        "inferred_pack_units",
        "price_per_unit",
        "subcategory_mean_price_per_unit",
    ]

    out_signals = signals.merge(sig[["product_id"] + derived_cols], on="product_id", how="left")

    # 3) Retrieval corpus refresh (rebuild from products to keep schema stable).
    retrieval_cols = ["product_id", "product_document", "category", "subcategory"]
    if price_col is not None and price_col in products.columns:
        retrieval_cols.append(price_col)
    if "avg_rating" in products.columns:
        retrieval_cols.append("avg_rating")
    retrieval = products[[c for c in retrieval_cols if c in products.columns]].copy()

    # Write outputs (in place by default to match the report’s semantics).
    if args.write_inplace:
        products.to_parquet(products_path, index=False)
        out_signals.to_parquet(signals_path, index=False)
        retrieval.to_parquet(retrieval_path, index=False)
    else:
        products.to_parquet(snap_dir / "products.enriched.parquet", index=False)
        out_signals.to_parquet(snap_dir / "product_signals.enriched.parquet", index=False)
        retrieval.to_parquet(snap_dir / "retrieval_corpus.enriched.parquet", index=False)

    sidecar = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": str(args.snapshot_id),
        "write_inplace": bool(args.write_inplace),
        "derived_columns_added_to_product_signals": derived_cols,
        "product_document_template": [
            "title",
            "brand",
            "category",
            "subcategory",
            "price_usd",
            "avg_rating",
            "rating_count",
            "description",
        ],
    }
    (snap_dir / "manifest_derived_enrich.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "snapshot_id": str(args.snapshot_id), "write_inplace": bool(args.write_inplace)}, indent=2))


if __name__ == "__main__":
    main()

