from __future__ import annotations

from typing import Iterable

import pandas as pd
from google.cloud import bigquery


def client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def ensure_dataset(project_id: str, dataset: str, location: str) -> None:
    bq = client(project_id)
    ds = bigquery.Dataset(f"{project_id}.{dataset}")
    ds.location = location
    bq.create_dataset(ds, exists_ok=True)


def run_sql(project_id: str, sql: str) -> None:
    bq = client(project_id)
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        bq.query(statement).result()


def drop_tables(project_id: str, dataset: str, table_names: Iterable[str]) -> None:
    bq = client(project_id)
    for name in table_names:
        bq.delete_table(f"{project_id}.{dataset}.{name}", not_found_ok=True)


def _normalize_timestamps_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid pyarrow errors loading ns-resolution datetimes to BigQuery (us, UTC)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            s = pd.to_datetime(out[col], utc=True)
            out[col] = s.dt.floor("us")
    return out


def load_df(
    project_id: str,
    dataset: str,
    table: str,
    df: pd.DataFrame,
    write_disposition: str = "WRITE_TRUNCATE",
    partition_field: str | None = None,
    clustering_fields: list[str] | None = None,
) -> None:
    if df.empty:
        return
    df = _normalize_timestamps_for_bq(df)
    cfg = bigquery.LoadJobConfig(write_disposition=write_disposition)
    if partition_field:
        cfg.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
    if clustering_fields:
        cfg.clustering_fields = clustering_fields
    bq = client(project_id)
    bq.load_table_from_dataframe(df, f"{project_id}.{dataset}.{table}", job_config=cfg).result()

