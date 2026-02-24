from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_utils import load_bq_dataframe, resolve_bq_settings, split_by_hash, write_json


def _save_split(df: pd.DataFrame, split_col: str, out_dir: Path, stem: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        sub = df[df[split_col] == split].drop(columns=[split_col]).reset_index(drop=True)
        sub.to_parquet(out_dir / f"{stem}_{split}.parquet", index=False)
        counts[split] = int(len(sub))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/data")
    parser.add_argument("--sentiment-max-rows", type=int, default=100000)
    parser.add_argument("--ranking-max-rows", type=int, default=500000)
    parser.add_argument("--pricing-max-rows", type=int, default=200000)
    args = parser.parse_args()

    project_id, dataset = resolve_bq_settings(args.project_id, args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentiment_sql = f"""
    SELECT product_id, text, label, ingested_at
    FROM `{project_id}.{dataset}.sentiment_dataset`
    WHERE text IS NOT NULL AND TRIM(text) != ''
    LIMIT {int(args.sentiment_max_rows)}
    """
    sentiment = load_bq_dataframe(project_id, sentiment_sql)
    sentiment["split"] = split_by_hash(sentiment, ["product_id", "text", "label"])
    sentiment_counts = _save_split(sentiment, "split", out_dir, "sentiment")

    ranking_sql = f"""
    SELECT
      rp.query_text,
      rp.product_id,
      rp.relevance_label,
      rc.product_document,
      rp.ingested_at
    FROM `{project_id}.{dataset}.ranking_pairs` rp
    JOIN `{project_id}.{dataset}.retrieval_corpus` rc
      ON rp.product_id = rc.product_id
    WHERE rp.query_text IS NOT NULL
      AND rc.product_document IS NOT NULL
      AND TRIM(rc.product_document) != ''
    LIMIT {int(args.ranking_max_rows)}
    """
    ranking = load_bq_dataframe(project_id, ranking_sql)
    ranking["split"] = split_by_hash(ranking, ["query_text", "product_id"])
    ranking_counts = _save_split(ranking, "split", out_dir, "ranking")

    pricing_sql = f"""
    SELECT
      product_id,
      price,
      avg_rating,
      review_count,
      positive_ratio,
      rating_price_ratio,
      target_price,
      ingested_at
    FROM `{project_id}.{dataset}.pricing_features`
    WHERE target_price IS NOT NULL
      AND target_price > 0
    LIMIT {int(args.pricing_max_rows)}
    """
    pricing = load_bq_dataframe(project_id, pricing_sql)
    pricing["split"] = split_by_hash(pricing, ["product_id"])
    pricing_counts = _save_split(pricing, "split", out_dir, "pricing")

    retrieval_sql = f"""
    SELECT product_id, product_document, price, avg_rating, ingested_at
    FROM `{project_id}.{dataset}.retrieval_corpus`
    WHERE product_document IS NOT NULL AND TRIM(product_document) != ''
    """
    retrieval = load_bq_dataframe(project_id, retrieval_sql)
    retrieval.to_parquet(out_dir / "retrieval_corpus.parquet", index=False)

    summary = {
        "project_id": project_id,
        "dataset": dataset,
        "sentiment_rows": int(len(sentiment)),
        "ranking_rows": int(len(ranking)),
        "pricing_rows": int(len(pricing)),
        "retrieval_rows": int(len(retrieval)),
        "splits": {
            "sentiment": sentiment_counts,
            "ranking": ranking_counts,
            "pricing": pricing_counts,
        },
        "label_distribution": {
            "sentiment": {str(k): int(v) for k, v in sentiment["label"].value_counts().to_dict().items()},
            "ranking": {str(k): int(v) for k, v in ranking["relevance_label"].value_counts().to_dict().items()},
        },
        "missing_values": {
            "sentiment_text_nulls": int(sentiment["text"].isna().sum()),
            "ranking_doc_nulls": int(ranking["product_document"].isna().sum()),
            "pricing_target_nulls": int(pricing["target_price"].isna().sum()),
        },
    }
    write_json(out_dir / "data_prep_summary.json", summary)
    print(f"Saved datasets and summary to {out_dir}")


if __name__ == "__main__":
    main()
