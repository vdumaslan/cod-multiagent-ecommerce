from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from prefect import flow, task

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from google.api_core import exceptions as gexc

from bq_utils import (
    dataset_exists,
    ensure_dataset,
    ensure_pipeline_runs_table,
    log_pipeline_event,
    run_sql,
)
from data_quality_report import build_report, validate_report
from ingest_sources import ingest_all
from pipeline_config import get_project_settings, get_split_settings, load_pipeline_config


@task
def setup_infra(project_id: str, dataset: str, location: str) -> None:
    try:
        ensure_dataset(project_id, dataset, location)
    except gexc.Forbidden as exc:
        # Some service accounts cannot create datasets.
        if not dataset_exists(project_id, dataset):
            raise RuntimeError(
                f"Dataset {project_id}.{dataset} does not exist and credentials cannot create datasets. "
                "Create it manually in BigQuery UI or grant bigquery.datasets.create permission."
            ) from exc
    ensure_pipeline_runs_table(project_id, dataset)


@task
def run_ingestion(config_path: str, project_id: str, dataset: str, max_rows: int | None) -> dict[str, int]:
    return ingest_all(config_path, project_id, dataset, max_rows=max_rows)


@task
def run_sql_templates(
    project_id: str,
    dataset: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    for fname in ("transform_bigquery.sql", "build_features.sql"):
        path = SCRIPT_DIR / fname
        sql = path.read_text(encoding="utf-8")
        sql = (
            sql.replace("{PROJECT_ID}", project_id)
            .replace("{DATASET}", dataset)
            .replace("{TRAIN_RATIO}", str(train_ratio))
            .replace("{VAL_RATIO}", str(val_ratio))
            .replace("{SEED}", str(seed))
        )
        run_sql(project_id, sql)


@task
def run_quality_checks(config_path: str, project_id: str, dataset: str) -> dict[str, object]:
    cfg = load_pipeline_config(config_path)
    report = build_report(project_id, dataset)
    errors = validate_report(report, cfg)
    payload: dict[str, object] = {"report": report, "errors": errors}
    out = Path("seller-copilot/artifacts/quality_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fail_on_error = bool(cfg.get("pipeline", {}).get("quality", {}).get("fail_on_error", True))
    if fail_on_error and errors:
        raise RuntimeError(f"Quality checks failed: {errors}")
    return payload


@flow(name="seller_copilot_data_pipeline")
def seller_copilot_pipeline(config_path: str = "seller-copilot/config/pipeline.yaml", max_rows: int | None = None) -> None:
    cfg = load_pipeline_config(config_path)
    project_id, location, dataset = get_project_settings(cfg)
    train, val, _test, seed = get_split_settings(cfg)
    run_id = str(uuid.uuid4())

    setup_infra(project_id, dataset, location)
    log_pipeline_event(project_id, dataset, run_id, "STARTED", "Pipeline started")
    try:
        ingest_stats = run_ingestion(config_path, project_id, dataset, max_rows)
        run_sql_templates(project_id, dataset, train, val, seed)
        quality = run_quality_checks(config_path, project_id, dataset)
        log_pipeline_event(
            project_id,
            dataset,
            run_id,
            "SUCCEEDED",
            f"Ingestion={ingest_stats}; quality_errors={quality.get('errors', [])}",
        )
    except Exception as exc:
        log_pipeline_event(project_id, dataset, run_id, "FAILED", str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="seller-copilot/config/pipeline.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    seller_copilot_pipeline(config_path=args.config, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
