from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.cloud import bigquery


def query_scalar(client: bigquery.Client, sql: str) -> int:
    row = list(client.query(sql).result())[0]
    return int(row[0])


def build_report(project_id: str, dataset: str) -> dict[str, int]:
    client = bigquery.Client(project=project_id)
    base = f"`{project_id}.{dataset}`"
    return {
        "stg_rows": query_scalar(client, f"SELECT COUNT(*) FROM {base}.stg_amazon_reviews"),
        "products_rows": query_scalar(client, f"SELECT COUNT(*) FROM {base}.products"),
        "reviews_rows": query_scalar(client, f"SELECT COUNT(*) FROM {base}.reviews"),
        "feature_rows": query_scalar(client, f"SELECT COUNT(*) FROM {base}.product_features"),
        "null_product_id_in_features": query_scalar(
            client, f"SELECT COUNT(*) FROM {base}.product_features WHERE product_id IS NULL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="seller-copilot/artifacts/quality_report.json")
    args = parser.parse_args()

    report = build_report(args.project_id, args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved quality report: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


