"""Validate a copilot-v2 snapshot locally or in BigQuery."""
from __future__ import annotations

import argparse
import json
from typing import Any

from scripts.cloud.cloud_bigquery import (
    BigQueryPipelineConfig,
    ensure_control_tables,
    load_bigquery_client,
    resolve_table_names,
    validate_bigquery_snapshot,
    validate_local_snapshot,
)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=None, help="GCP project id. Defaults to GCP_PROJECT_ID.")
    parser.add_argument("--dataset", default=None, help="BigQuery dataset id. Defaults to BIGQUERY_DATASET or copilot_v2.")
    parser.add_argument("--location", default=None, help="BigQuery location. Defaults to BIGQUERY_LOCATION or US.")
    parser.add_argument("--snapshot-id", default=None, help="Artifact snapshot id.")
    parser.add_argument("--artifacts-root", default="copilot-v2/artifacts", help="Path to copilot-v2/artifacts.")
    parser.add_argument(
        "--table-set",
        default="serving",
        choices=["caches", "serving", "full"],
        help="Validation scope.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Validate a specific table id. Can be repeated; overrides --table-set.",
    )
    parser.add_argument("--local-only", action="store_true", help="Validate local artifact files only.")
    parser.add_argument("--run-id", default="", help="Pipeline run id to attach to written DQ rows.")
    parser.add_argument("--write-results", action="store_true", help="Append checks to data_quality_results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BigQueryPipelineConfig.from_env(
        project_id=args.project_id,
        dataset=args.dataset,
        location=args.location,
        snapshot_id=args.snapshot_id,
        artifacts_root=args.artifacts_root,
    )
    table_names = resolve_table_names(args.table_set, args.tables)

    if args.local_only:
        report = validate_local_snapshot(cfg=cfg, table_names=table_names)
    else:
        client = load_bigquery_client(cfg)
        if args.write_results:
            ensure_control_tables(client, cfg)
        report = validate_bigquery_snapshot(
            client=client,
            cfg=cfg,
            table_names=table_names,
            run_id=args.run_id or None,
            write_results=bool(args.write_results),
        )

    print(_json(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

