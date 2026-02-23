from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from google.cloud import bigquery

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import load_pipeline_config


def query_scalar(client: bigquery.Client, sql: str) -> int:
    rows = list(client.query(sql).result())
    return int(rows[0][0]) if rows else 0


def build_report(project_id: str, dataset: str) -> dict[str, int]:
    client = bigquery.Client(project=project_id)
    base = f"`{project_id}.{dataset}`"
    keys = [
        "stg_amazon_reviews",
        "stg_twitter_support",
        "stg_online_retail",
        "stg_telco_churn",
        "products",
        "reviews",
        "product_features",
        "sentiment_dataset",
        "ranking_pairs",
        "pricing_features",
    ]
    report: dict[str, int] = {}
    for key in keys:
        report[key] = query_scalar(client, f"SELECT COUNT(*) FROM {base}.{key}")
    report["null_product_id_in_product_features"] = query_scalar(
        client, f"SELECT COUNT(*) FROM {base}.product_features WHERE product_id IS NULL"
    )
    return report


def validate_report(report: dict[str, int], cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    mins = cfg.get("pipeline", {}).get("quality", {}).get("min_rows", {})
    for table_name, min_rows in mins.items():
        actual = int(report.get(table_name, 0))
        if actual < int(min_rows):
            errors.append(f"{table_name} rows={actual} below min_rows={min_rows}")
    if report.get("null_product_id_in_product_features", 0) > 0:
        errors.append("product_features contains NULL product_id values")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="seller-copilot/config/pipeline.yaml")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="seller-copilot/artifacts/quality_report.json")
    parser.add_argument("--fail-on-error", choices=["true", "false"], default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    report = build_report(args.project_id, args.dataset)
    errors = validate_report(report, cfg)

    out_payload = {"report": report, "errors": errors}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(json.dumps(out_payload, indent=2))

    default_fail = bool(cfg.get("pipeline", {}).get("quality", {}).get("fail_on_error", True))
    fail_on_error = default_fail if args.fail_on_error is None else args.fail_on_error == "true"
    if fail_on_error and errors:
        raise RuntimeError("Quality checks failed. See quality report.")


if __name__ == "__main__":
    main()

