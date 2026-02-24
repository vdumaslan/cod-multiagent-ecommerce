from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import load_bq_dataframe, resolve_bq_settings, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="seller-copilot/artifacts/data_analytics_summary.json")
    args = parser.parse_args()

    project_id, dataset = resolve_bq_settings(args.project_id, args.dataset)

    sql = f"""
    WITH tables AS (
      SELECT 'products' AS table_name, COUNT(*) AS row_count FROM `{project_id}.{dataset}.products`
      UNION ALL SELECT 'reviews', COUNT(*) FROM `{project_id}.{dataset}.reviews`
      UNION ALL SELECT 'support_tickets', COUNT(*) FROM `{project_id}.{dataset}.support_tickets`
      UNION ALL SELECT 'product_features', COUNT(*) FROM `{project_id}.{dataset}.product_features`
      UNION ALL SELECT 'sentiment_dataset', COUNT(*) FROM `{project_id}.{dataset}.sentiment_dataset`
      UNION ALL SELECT 'ranking_pairs', COUNT(*) FROM `{project_id}.{dataset}.ranking_pairs`
      UNION ALL SELECT 'pricing_features', COUNT(*) FROM `{project_id}.{dataset}.pricing_features`
      UNION ALL SELECT 'retrieval_corpus', COUNT(*) FROM `{project_id}.{dataset}.retrieval_corpus`
    )
    SELECT * FROM tables
    """
    tables = load_bq_dataframe(project_id, sql)

    rating_sql = f"""
    SELECT
      ROUND(AVG(rating), 4) AS avg_rating,
      COUNTIF(rating >= 4) AS positive_reviews,
      COUNT(*) AS total_reviews
    FROM `{project_id}.{dataset}.reviews`
    """
    rating = load_bq_dataframe(project_id, rating_sql).iloc[0].to_dict()

    price_sql = f"""
    SELECT
      ROUND(AVG(price), 4) AS avg_price,
      ROUND(APPROX_QUANTILES(price, 10)[OFFSET(5)], 4) AS median_price,
      ROUND(MAX(price), 4) AS max_price
    FROM `{project_id}.{dataset}.pricing_features`
    WHERE price IS NOT NULL
    """
    price = load_bq_dataframe(project_id, price_sql).iloc[0].to_dict()

    payload = {
        "project_id": project_id,
        "dataset": dataset,
        "table_row_counts": tables.to_dict(orient="records"),
        "review_stats": rating,
        "pricing_stats": price,
    }

    out = Path(args.output)
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
