"""Upload a copilot-v2 artifact snapshot into BigQuery.

Default mode is a dry run so it can be used safely on machines without GCP
credentials. Pass --no-dry-run to perform the upload.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.cloud.cloud_bigquery import (
    BigQueryPipelineConfig,
    ensure_control_tables,
    ensure_dataset,
    load_bigquery_client,
    local_table_plan,
    new_run_id,
    record_pipeline_run,
    resolve_table_names,
    upload_snapshot_table,
    validate_bigquery_snapshot,
    TABLE_SPECS,
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
        help="caches=runtime caches only, serving=runtime warehouse, full=includes large reviews/sales/splits.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Upload a specific table id. Can be repeated; overrides --table-set.",
    )
    dry = parser.add_mutually_exclusive_group()
    dry.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Plan only; do not contact GCP.")
    dry.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Actually create/load BigQuery tables.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip BigQuery validation after upload.")
    parser.add_argument("--write-validation-results", action="store_true", help="Append validation checks to data_quality_results.")
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
    plan = local_table_plan(cfg, table_names)
    missing = [p for p in plan if not p["exists"]]

    header = {
        "ok": not missing,
        "dry_run": bool(args.dry_run),
        "project_id": cfg.project_id or None,
        "dataset": cfg.dataset,
        "location": cfg.location,
        "snapshot_id": cfg.snapshot_id,
        "table_set": args.table_set,
        "tables": table_names,
        "plan": plan,
    }
    if args.dry_run or missing:
        print(_json(header))
        return 1 if missing else 0

    if not cfg.project_id:
        print(_json({"ok": False, "error": "missing_project_id", "message": "Set GCP_PROJECT_ID or pass --project-id."}))
        return 2

    run_id = new_run_id("bq_upload")
    started_at = None
    loaded: list[dict[str, Any]] = []
    client = load_bigquery_client(cfg)
    ensure_dataset(client, cfg)
    ensure_control_tables(client, cfg)

    try:
        from scripts.cloud.cloud_bigquery import utc_now_iso

        started_at = utc_now_iso()
        for name in table_names:
            result = upload_snapshot_table(
                client=client,
                cfg=cfg,
                spec=TABLE_SPECS[name],
                run_id=run_id,
            )
            loaded.append(result)

        validation_report = None
        if not args.skip_validation:
            validation_report = validate_bigquery_snapshot(
                client=client,
                cfg=cfg,
                table_names=table_names,
                run_id=run_id,
                write_results=bool(args.write_validation_results),
            )
            if not validation_report["ok"]:
                raise RuntimeError("BigQuery validation failed")

        finished_at = utc_now_iso()
        details = {"loaded": loaded, "validation": validation_report}
        record_pipeline_run(
            client=client,
            cfg=cfg,
            run_id=run_id,
            table_set=args.table_set,
            status="SUCCESS",
            started_at=started_at,
            finished_at=finished_at,
            tables_loaded=len(loaded),
            dry_run=False,
            details=details,
        )
        print(_json({"ok": True, "run_id": run_id, "loaded": loaded, "validation": validation_report}))
        return 0
    except Exception as exc:
        try:
            from scripts.cloud.cloud_bigquery import utc_now_iso

            record_pipeline_run(
                client=client,
                cfg=cfg,
                run_id=run_id,
                table_set=args.table_set,
                status="FAILED",
                started_at=started_at or utc_now_iso(),
                finished_at=utc_now_iso(),
                tables_loaded=len(loaded),
                dry_run=False,
                details={"loaded": loaded, "error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        print(_json({"ok": False, "run_id": run_id, "loaded": loaded, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

