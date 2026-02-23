from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from google.cloud import bigquery
from google.api_core import exceptions as gexc


def get_client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def ensure_dataset(project_id: str, dataset_id: str, location: str) -> None:
    client = get_client(project_id)
    full_id = f"{project_id}.{dataset_id}"
    dataset = bigquery.Dataset(full_id)
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)


def dataset_exists(project_id: str, dataset_id: str) -> bool:
    client = get_client(project_id)
    full_id = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(full_id)
        return True
    except gexc.NotFound:
        return False


def ensure_pipeline_runs_table(project_id: str, dataset_id: str) -> None:
    client = get_client(project_id)
    table_id = f"{project_id}.{dataset_id}.pipeline_runs"
    schema = [
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("event_ts", "TIMESTAMP"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("message", "STRING"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table, exists_ok=True)


def log_pipeline_event(project_id: str, dataset_id: str, run_id: str, status: str, message: str) -> None:
    client = get_client(project_id)
    table_id = f"{project_id}.{dataset_id}.pipeline_runs"
    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "event_ts": datetime.now(timezone.utc),
                "status": status,
                "message": message[:1024],
            }
        ]
    )
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_id, job_config=cfg)
    job.result()


def load_dataframe(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_name: str,
    partition_field: str | None = "ingested_at",
    clustering_fields: Iterable[str] | None = None,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    client = get_client(project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    cfg = bigquery.LoadJobConfig(write_disposition=write_disposition)
    if partition_field:
        cfg.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
    if clustering_fields:
        cfg.clustering_fields = list(clustering_fields)
    job = client.load_table_from_dataframe(df, table_id, job_config=cfg)
    job.result()


def run_sql(project_id: str, sql: str) -> None:
    client = get_client(project_id)
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        client.query(statement).result()
