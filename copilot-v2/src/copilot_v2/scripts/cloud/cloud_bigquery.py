"""BigQuery pipeline primitives for the copilot-v2 artifact snapshot.

The module is intentionally dependency-light at import time. BigQuery clients
are imported only when a non-dry-run command actually needs cloud access.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SNAPSHOT_ID = "38710839ca6e1009"
DEFAULT_DATASET = "copilot_v2"
DEFAULT_LOCATION = "US"


@dataclass(frozen=True)
class BigQueryPipelineConfig:
    project_id: str
    dataset: str
    location: str
    snapshot_id: str
    artifacts_root: Path

    @classmethod
    def from_env(
        cls,
        *,
        project_id: str | None = None,
        dataset: str | None = None,
        location: str | None = None,
        snapshot_id: str | None = None,
        artifacts_root: str | Path | None = None,
    ) -> "BigQueryPipelineConfig":
        project = (
            project_id
            or os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or ""
        ).strip()
        ds = (dataset or os.environ.get("BIGQUERY_DATASET") or DEFAULT_DATASET).strip()
        if "." in ds:
            ds_project, ds_id = ds.split(".", 1)
            project = project or ds_project
            ds = ds_id
        loc = (location or os.environ.get("BIGQUERY_LOCATION") or DEFAULT_LOCATION).strip()
        snap = (snapshot_id or os.environ.get("COPILOT_SNAPSHOT_ID") or DEFAULT_SNAPSHOT_ID).strip()
        root = Path(artifacts_root or os.environ.get("COPILOT_ARTIFACTS_ROOT") or "copilot-v2/artifacts")
        return cls(project_id=project, dataset=ds, location=loc, snapshot_id=snap, artifacts_root=root)


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    relative_path: str
    description: str
    expected_rows: int | None = None
    unique_key: str | None = None
    cluster_fields: tuple[str, ...] = ("snapshot_id", "product_id")
    serving: bool = False
    full: bool = False

    def local_path(self, cfg: BigQueryPipelineConfig) -> Path:
        return cfg.artifacts_root / self.relative_path.format(snapshot_id=cfg.snapshot_id)


TABLE_SPECS: dict[str, TableSpec] = {
    "product_metadata": TableSpec(
        table_id="product_metadata",
        relative_path="data_snapshots/{snapshot_id}/products.parquet",
        description="Cleaned product catalog used by runtime agents.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "reviews": TableSpec(
        table_id="reviews",
        relative_path="data_snapshots/{snapshot_id}/reviews.parquet",
        description="Filtered review records. Large table; needed for full warehouse rebuilds.",
        expected_rows=5_310_178,
        unique_key="review_id",
        cluster_fields=("snapshot_id", "product_id"),
        full=True,
    ),
    "product_signals": TableSpec(
        table_id="product_signals",
        relative_path="data_snapshots/{snapshot_id}/product_signals.parquet",
        description="Product-level review, rating, and price-derived signals.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "product_features": TableSpec(
        table_id="product_features",
        relative_path="features/{snapshot_id}/tabular_features.parquet",
        description="Engineered tabular features used by pricing and audit workflows.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "retrieval_corpus": TableSpec(
        table_id="retrieval_corpus",
        relative_path="data_snapshots/{snapshot_id}/retrieval_corpus.parquet",
        description="Product documents used to build and audit the FAISS retrieval index.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "products_split": TableSpec(
        table_id="products_split",
        relative_path="splits/{snapshot_id}/products_split.parquet",
        description="Product train/validation/test split.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=False,
        full=True,
    ),
    "reviews_split": TableSpec(
        table_id="reviews_split",
        relative_path="splits/{snapshot_id}/reviews_split.parquet",
        description="Review train/validation/test split.",
        expected_rows=5_310_178,
        unique_key="review_id",
        cluster_fields=("snapshot_id", "product_id"),
        serving=False,
        full=True,
    ),
    "pricing_cache": TableSpec(
        table_id="pricing_cache",
        relative_path="caches/{snapshot_id}/pricing/pricing_cache.parquet",
        description="Precomputed TabPFN pricing recommendations.",
        expected_rows=35_259,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "sentiment_cache": TableSpec(
        table_id="sentiment_cache",
        relative_path="caches/{snapshot_id}/sentiment/sentiment_cache.parquet",
        description="Precomputed DistilRoBERTa sentiment aggregates.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "inventory_cache": TableSpec(
        table_id="inventory_cache",
        relative_path="caches/{snapshot_id}/inventory/inventory_cache.parquet",
        description="Precomputed inventory classification cache.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "inventory_skus": TableSpec(
        table_id="inventory_skus",
        relative_path="synthetic/{snapshot_id}/inventory_skus.parquet",
        description="Synthetic SKU-level inventory inputs.",
        expected_rows=50_000,
        unique_key="product_id",
        serving=True,
        full=True,
    ),
    "sales_daily": TableSpec(
        table_id="sales_daily",
        relative_path="synthetic/{snapshot_id}/sales_daily.parquet",
        description="Synthetic daily sales/returns fact table. Large table; full warehouse only.",
        expected_rows=4_500_000,
        unique_key=None,
        cluster_fields=("snapshot_id", "product_id"),
        serving=False,
        full=True,
    ),
}

TABLE_SETS: dict[str, tuple[str, ...]] = {
    "caches": ("pricing_cache", "sentiment_cache", "inventory_cache"),
    "serving": tuple(k for k, v in TABLE_SPECS.items() if v.serving),
    "full": tuple(k for k, v in TABLE_SPECS.items() if v.full),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def sanitize_table_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value))[:900]


def resolve_table_names(table_set: str, explicit_tables: list[str] | None = None) -> list[str]:
    if explicit_tables:
        unknown = sorted(set(explicit_tables) - set(TABLE_SPECS))
        if unknown:
            raise ValueError(f"Unknown table(s): {', '.join(unknown)}")
        return list(dict.fromkeys(explicit_tables))
    key = str(table_set or "serving").strip().lower()
    if key not in TABLE_SETS:
        raise ValueError(f"Unknown table_set={table_set!r}; expected one of {sorted(TABLE_SETS)}")
    return list(TABLE_SETS[key])


def qualified_table(cfg: BigQueryPipelineConfig, table_id: str) -> str:
    if not cfg.project_id:
        raise ValueError("GCP project is required for BigQuery operations")
    return f"{cfg.project_id}.{cfg.dataset}.{table_id}"


def backtick_table(cfg: BigQueryPipelineConfig, table_id: str) -> str:
    return f"`{qualified_table(cfg, table_id)}`"


def load_bigquery_client(cfg: BigQueryPipelineConfig) -> Any:
    if not cfg.project_id:
        raise ValueError("Set GCP_PROJECT_ID or pass --project-id before running BigQuery operations")
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install google-cloud-bigquery to use BigQuery mode") from exc
    return bigquery.Client(project=cfg.project_id, location=cfg.location)


def ensure_dataset(client: Any, cfg: BigQueryPipelineConfig) -> None:
    from google.cloud import bigquery  # type: ignore

    dataset_ref = f"{cfg.project_id}.{cfg.dataset}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = cfg.location
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(dataset, exists_ok=True)


def ensure_control_tables(client: Any, cfg: BigQueryPipelineConfig) -> None:
    control_sql = [
        f"""
        CREATE TABLE IF NOT EXISTS {backtick_table(cfg, "pipeline_runs")} (
          run_id STRING,
          snapshot_id STRING,
          table_set STRING,
          status STRING,
          started_at TIMESTAMP,
          finished_at TIMESTAMP,
          tables_loaded INT64,
          dry_run BOOL,
          details_json STRING
        )
        PARTITION BY DATE(started_at)
        CLUSTER BY snapshot_id, status
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {backtick_table(cfg, "data_quality_results")} (
          run_id STRING,
          snapshot_id STRING,
          table_id STRING,
          check_name STRING,
          status STRING,
          observed_value FLOAT64,
          expected_value FLOAT64,
          details_json STRING,
          checked_at TIMESTAMP
        )
        PARTITION BY DATE(checked_at)
        CLUSTER BY snapshot_id, table_id, status
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {backtick_table(cfg, "operator_decision_log")} (
          event_ts TIMESTAMP,
          snapshot_id STRING,
          owner_id STRING,
          run_id STRING,
          variant STRING,
          event STRING,
          goal STRING,
          product_id STRING,
          accepted BOOL,
          confidence_rating INT64,
          metadata_json STRING
        )
        PARTITION BY DATE(event_ts)
        CLUSTER BY snapshot_id, owner_id, event
        """,
    ]
    for sql in control_sql:
        client.query(sql, location=cfg.location).result()


def table_exists(client: Any, cfg: BigQueryPipelineConfig, table_id: str) -> bool:
    try:
        client.get_table(qualified_table(cfg, table_id))
        return True
    except Exception:
        return False


def delete_table_if_exists(client: Any, cfg: BigQueryPipelineConfig, table_id: str) -> None:
    try:
        client.delete_table(qualified_table(cfg, table_id), not_found_ok=True)
    except Exception:
        pass


def parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        return {
            "rows": int(pf.metadata.num_rows),
            "columns": list(pf.schema_arrow.names),
            "row_groups": int(pf.metadata.num_row_groups),
            "bytes": int(path.stat().st_size),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "bytes": int(path.stat().st_size) if path.exists() else 0}


def local_table_plan(cfg: BigQueryPipelineConfig, table_names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in table_names:
        spec = TABLE_SPECS[name]
        path = spec.local_path(cfg)
        item: dict[str, Any] = {
            "table_id": spec.table_id,
            "description": spec.description,
            "path": str(path),
            "exists": path.is_file(),
            "expected_rows": spec.expected_rows,
        }
        if path.is_file():
            item.update(parquet_metadata(path))
        out.append(item)
    return out


def _query_params(snapshot_id: str) -> Any:
    from google.cloud import bigquery  # type: ignore

    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("snapshot_id", "STRING", snapshot_id)]
    )


def _load_local_parquet_to_temp(
    *,
    client: Any,
    cfg: BigQueryPipelineConfig,
    source_path: Path,
    temp_table_id: str,
) -> int:
    from google.cloud import bigquery  # type: ignore

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with source_path.open("rb") as f:
        job = client.load_table_from_file(
            f,
            qualified_table(cfg, temp_table_id),
            location=cfg.location,
            job_config=job_config,
        )
    job.result()
    return int(client.get_table(qualified_table(cfg, temp_table_id)).num_rows)


def _cluster_clause(spec: TableSpec) -> str:
    fields = [f for f in spec.cluster_fields if f]
    if not fields:
        return ""
    return "CLUSTER BY " + ", ".join(fields)


def upload_snapshot_table(
    *,
    client: Any,
    cfg: BigQueryPipelineConfig,
    spec: TableSpec,
    run_id: str,
) -> dict[str, Any]:
    source_path = spec.local_path(cfg)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing local artifact for {spec.table_id}: {source_path}")

    safe_run = sanitize_table_token(run_id)
    temp_source = f"_tmp_{spec.table_id}_source_{safe_run}"
    temp_next = f"_tmp_{spec.table_id}_next_{safe_run}"
    target = spec.table_id
    started = utc_now_iso()

    rows_loaded = _load_local_parquet_to_temp(
        client=client,
        cfg=cfg,
        source_path=source_path,
        temp_table_id=temp_source,
    )

    part_cluster = f"PARTITION BY DATE(ingested_at) {_cluster_clause(spec)}"
    new_snapshot_select = (
        f"SELECT @snapshot_id AS snapshot_id, CURRENT_TIMESTAMP() AS ingested_at, "
        f"t.* FROM {backtick_table(cfg, temp_source)} AS t"
    )
    if table_exists(client, cfg, target):
        next_sql = f"""
        CREATE OR REPLACE TABLE {backtick_table(cfg, temp_next)} AS
        SELECT * FROM {backtick_table(cfg, target)}
        WHERE snapshot_id != @snapshot_id
        UNION ALL
        {new_snapshot_select}
        """
    else:
        next_sql = f"""
        CREATE OR REPLACE TABLE {backtick_table(cfg, temp_next)} AS
        {new_snapshot_select}
        """
    client.query(next_sql, location=cfg.location, job_config=_query_params(cfg.snapshot_id)).result()

    final_sql = f"""
    CREATE OR REPLACE TABLE {backtick_table(cfg, target)}
    {part_cluster}
    AS
    SELECT * FROM {backtick_table(cfg, temp_next)}
    """
    client.query(final_sql, location=cfg.location).result()

    delete_table_if_exists(client, cfg, temp_source)
    delete_table_if_exists(client, cfg, temp_next)
    finished = utc_now_iso()
    return {
        "table_id": spec.table_id,
        "source_path": str(source_path),
        "rows_loaded": rows_loaded,
        "started_at": started,
        "finished_at": finished,
    }


def append_json_rows(client: Any, cfg: BigQueryPipelineConfig, table_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    from google.cloud import bigquery  # type: ignore

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_json(
        rows,
        qualified_table(cfg, table_id),
        location=cfg.location,
        job_config=job_config,
    ).result()


def record_pipeline_run(
    *,
    client: Any,
    cfg: BigQueryPipelineConfig,
    run_id: str,
    table_set: str,
    status: str,
    started_at: str,
    finished_at: str,
    tables_loaded: int,
    dry_run: bool,
    details: dict[str, Any],
) -> None:
    append_json_rows(
        client,
        cfg,
        "pipeline_runs",
        [
            {
                "run_id": run_id,
                "snapshot_id": cfg.snapshot_id,
                "table_set": table_set,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "tables_loaded": tables_loaded,
                "dry_run": dry_run,
                "details_json": json.dumps(details, sort_keys=True),
            }
        ],
    )


def _read_product_ids(path: Path, column: str = "product_id") -> set[str]:
    import pandas as pd

    df = pd.read_parquet(path, columns=[column])
    return set(df[column].astype(str).tolist())


def validate_local_snapshot(
    *,
    cfg: BigQueryPipelineConfig,
    table_names: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    plan = local_table_plan(cfg, table_names)

    for item in plan:
        status = "PASS" if item["exists"] else "FAIL"
        checks.append(
            {
                "table_id": item["table_id"],
                "check_name": "local_file_exists",
                "status": status,
                "observed_value": 1.0 if item["exists"] else 0.0,
                "expected_value": 1.0,
                "details": {"path": item["path"]},
            }
        )
        if not item["exists"]:
            continue
        expected = item.get("expected_rows")
        observed = item.get("rows")
        if expected is not None and observed is not None:
            checks.append(
                {
                    "table_id": item["table_id"],
                    "check_name": "row_count",
                    "status": "PASS" if int(observed) == int(expected) else "FAIL",
                    "observed_value": float(observed),
                    "expected_value": float(expected),
                    "details": {},
                }
            )

    product_path = TABLE_SPECS["product_metadata"].local_path(cfg)
    product_ids: set[str] | None = None
    if product_path.is_file():
        product_ids = _read_product_ids(product_path)

    for name in table_names:
        spec = TABLE_SPECS[name]
        path = spec.local_path(cfg)
        if not path.is_file() or not spec.unique_key:
            continue
        try:
            ids = _read_product_ids(path, spec.unique_key)
            meta = parquet_metadata(path)
            rows = int(meta.get("rows") or 0)
            checks.append(
                {
                    "table_id": spec.table_id,
                    "check_name": f"unique_{spec.unique_key}",
                    "status": "PASS" if len(ids) == rows else "FAIL",
                    "observed_value": float(len(ids)),
                    "expected_value": float(rows),
                    "details": {},
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "table_id": spec.table_id,
                    "check_name": f"unique_{spec.unique_key}",
                    "status": "FAIL",
                    "observed_value": 0.0,
                    "expected_value": 1.0,
                    "details": {"error": f"{type(exc).__name__}: {exc}"},
                }
            )

    if product_ids:
        for name in ["pricing_cache", "sentiment_cache", "inventory_cache", "retrieval_corpus", "product_features"]:
            if name not in table_names:
                continue
            path = TABLE_SPECS[name].local_path(cfg)
            if not path.is_file():
                continue
            ids = _read_product_ids(path)
            outside = len(ids - product_ids)
            checks.append(
                {
                    "table_id": name,
                    "check_name": "product_id_subset_of_product_metadata",
                    "status": "PASS" if outside == 0 else "FAIL",
                    "observed_value": float(outside),
                    "expected_value": 0.0,
                    "details": {},
                }
            )

    return {
        "ok": all(c["status"] == "PASS" for c in checks),
        "snapshot_id": cfg.snapshot_id,
        "mode": "local",
        "checked_at": utc_now_iso(),
        "checks": checks,
        "plan": plan,
    }


def _scalar(client: Any, cfg: BigQueryPipelineConfig, sql: str) -> Any:
    rows = list(client.query(sql, location=cfg.location, job_config=_query_params(cfg.snapshot_id)).result())
    if not rows:
        return None
    return list(dict(rows[0]).values())[0]


def _table_count(client: Any, cfg: BigQueryPipelineConfig, table_id: str) -> int:
    sql = f"SELECT COUNT(1) AS n FROM {backtick_table(cfg, table_id)} WHERE snapshot_id = @snapshot_id"
    return int(_scalar(client, cfg, sql) or 0)


def _distinct_count(client: Any, cfg: BigQueryPipelineConfig, table_id: str, column: str) -> int:
    sql = (
        f"SELECT COUNT(DISTINCT `{column}`) AS n "
        f"FROM {backtick_table(cfg, table_id)} WHERE snapshot_id = @snapshot_id"
    )
    return int(_scalar(client, cfg, sql) or 0)


def validate_bigquery_snapshot(
    *,
    client: Any,
    cfg: BigQueryPipelineConfig,
    table_names: list[str],
    run_id: str | None = None,
    write_results: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checked_at = utc_now_iso()

    for name in table_names:
        spec = TABLE_SPECS[name]
        exists = table_exists(client, cfg, spec.table_id)
        checks.append(
            {
                "table_id": spec.table_id,
                "check_name": "bigquery_table_exists",
                "status": "PASS" if exists else "FAIL",
                "observed_value": 1.0 if exists else 0.0,
                "expected_value": 1.0,
                "details": {},
            }
        )
        if not exists:
            continue

        rows = _table_count(client, cfg, spec.table_id)
        if spec.expected_rows is not None:
            checks.append(
                {
                    "table_id": spec.table_id,
                    "check_name": "row_count",
                    "status": "PASS" if rows == spec.expected_rows else "FAIL",
                    "observed_value": float(rows),
                    "expected_value": float(spec.expected_rows),
                    "details": {},
                }
            )
        if spec.unique_key:
            distinct = _distinct_count(client, cfg, spec.table_id, spec.unique_key)
            checks.append(
                {
                    "table_id": spec.table_id,
                    "check_name": f"unique_{spec.unique_key}",
                    "status": "PASS" if distinct == rows else "FAIL",
                    "observed_value": float(distinct),
                    "expected_value": float(rows),
                    "details": {},
                }
            )

    if table_exists(client, cfg, "product_metadata"):
        for table_id in ["pricing_cache", "sentiment_cache", "inventory_cache", "retrieval_corpus", "product_features"]:
            if table_id not in table_names or not table_exists(client, cfg, table_id):
                continue
            sql = f"""
            SELECT COUNT(1) AS n
            FROM {backtick_table(cfg, table_id)} AS t
            LEFT JOIN {backtick_table(cfg, "product_metadata")} AS p
              ON t.snapshot_id = p.snapshot_id AND t.product_id = p.product_id
            WHERE t.snapshot_id = @snapshot_id AND p.product_id IS NULL
            """
            outside = int(_scalar(client, cfg, sql) or 0)
            checks.append(
                {
                    "table_id": table_id,
                    "check_name": "product_id_subset_of_product_metadata",
                    "status": "PASS" if outside == 0 else "FAIL",
                    "observed_value": float(outside),
                    "expected_value": 0.0,
                    "details": {},
                }
            )

    ok = all(c["status"] == "PASS" for c in checks)
    report = {
        "ok": ok,
        "snapshot_id": cfg.snapshot_id,
        "mode": "bigquery",
        "checked_at": checked_at,
        "checks": checks,
    }

    if write_results:
        ensure_control_tables(client, cfg)
        rows = []
        for c in checks:
            rows.append(
                {
                    "run_id": run_id or "",
                    "snapshot_id": cfg.snapshot_id,
                    "table_id": c["table_id"],
                    "check_name": c["check_name"],
                    "status": c["status"],
                    "observed_value": c.get("observed_value"),
                    "expected_value": c.get("expected_value"),
                    "details_json": json.dumps(c.get("details") or {}, sort_keys=True),
                    "checked_at": checked_at,
                }
            )
        append_json_rows(client, cfg, "data_quality_results", rows)

    return report


def new_run_id(prefix: str = "bq") -> str:
    return f"{prefix}_{utc_now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
