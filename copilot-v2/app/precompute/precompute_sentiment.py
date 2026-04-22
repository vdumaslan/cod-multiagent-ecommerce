"""Batch sentiment inference over reviews → sentiment cache.

Winner: distilroberta-base (fine-tuned, 500k train rows)
Label buckets: neg=<=2 stars, neu==3, pos>=4
Source: artifacts/data_snapshots/{snapshot_id}/reviews.parquet
Output: artifacts/caches/{snapshot_id}/sentiment/{sentiment_cache.json, .parquet, _manifest.json}
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "38710839ca6e1009"
MODEL_NAME = "distilroberta-base"
LABEL_BUCKETS = {"neg": "<=2", "neu": "==3", "pos": ">=4"}
BATCH_SIZE = 256
MAX_LENGTH = 256


def _rating_to_label(rating: float) -> str:
    if rating <= 2:
        return "neg"
    if rating == 3:
        return "neu"
    return "pos"


def _load_model(checkpoint_path: Path | None):
    from transformers import pipeline

    model_path = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else MODEL_NAME
    print(f"Loading model from: {model_path}")
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        truncation=True,
        max_length=MAX_LENGTH,
        device=-1,
    )


def _aggregate(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby("product_id")[label_col]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
        .reset_index()
    )
    n_reviews = df.groupby("product_id").size().reset_index(name="n_reviews")
    result = n_reviews.merge(grouped, on="product_id", how="left")
    for col in ("neg", "neu", "pos"):
        if col not in result.columns:
            result[col] = 0
    total = result[["neg", "neu", "pos"]].sum(axis=1).replace(0, 1)
    result["p_pos"] = result["pos"] / total
    result["p_neu"] = result["neu"] / total
    result["p_neg"] = result["neg"] / total
    return result[["product_id", "n_reviews", "p_pos", "p_neu", "p_neg"]]


def run(
    snapshot_id: str = SNAPSHOT_ID,
    artifacts_root: Path | None = None,
    max_reviews: int = 0,
    use_model: bool = True,
) -> Path:
    root = artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
    src = root / "data_snapshots" / snapshot_id / "reviews.parquet"
    checkpoint = root / "models" / snapshot_id / "sentiment" / "distilroberta-base_final_500k_slice_winner"
    out_dir = root / "caches" / snapshot_id / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src}")
    df = pd.read_parquet(src, columns=["product_id", "review_text", "rating"])
    if max_reviews > 0:
        df = df.head(max_reviews)
    df["review_text"] = df["review_text"].fillna("").astype(str)

    if use_model:
        clf = _load_model(checkpoint)
        texts = df["review_text"].tolist()
        labels: list[str] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results = clf(batch, truncation=True, max_length=MAX_LENGTH)
            for r in results:
                raw = r["label"].lower()
                if "neg" in raw:
                    labels.append("neg")
                elif "neu" in raw or "neutral" in raw:
                    labels.append("neu")
                else:
                    labels.append("pos")
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  {i + len(batch)}/{len(texts)} reviews processed")
        df["label"] = labels
    else:
        df["label"] = df["rating"].apply(_rating_to_label)

    agg = _aggregate(df, "label")

    cache_dict = {
        row["product_id"]: {
            "n_reviews": float(row["n_reviews"]),
            "p_pos": float(row["p_pos"]),
            "p_neu": float(row["p_neu"]),
            "p_neg": float(row["p_neg"]),
        }
        for _, row in agg.iterrows()
    }

    (out_dir / "sentiment_cache.json").write_text(json.dumps(cache_dict), encoding="utf-8")
    agg.to_parquet(out_dir / "sentiment_cache.parquet", index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "source_reviews": str(src.as_posix()).split("cod-multiagent-ecommerce/")[-1],
        "max_reviews": max_reviews,
        "rows": len(agg),
        "columns": list(agg.columns),
        "approach": "distilroberta",
        "label_buckets": LABEL_BUCKETS,
    }
    (out_dir / "sentiment_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(agg)} products → {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default=SNAPSHOT_ID)
    parser.add_argument("--max-reviews", type=int, default=0)
    parser.add_argument("--no-model", action="store_true", help="Use star ratings instead of model inference")
    args = parser.parse_args()
    run(snapshot_id=args.snapshot_id, max_reviews=args.max_reviews, use_model=not args.no_model)
