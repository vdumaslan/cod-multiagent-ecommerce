"""Lazy BigQuery cache loading for the FastAPI runtime.

Runtime still serves from in-memory dictionaries. BigQuery is only used during
startup hydration when COPILOT_DATA_BACKEND=bigquery.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_DATASET = "copilot_v2"
DEFAULT_LOCATION = "US"


class BigQueryCacheError(RuntimeError):
    """Raised when BigQuery cache hydration cannot be completed."""


@dataclass(frozen=True)
class BigQueryCacheConfig:
    project_id: str
    dataset: str
    location: str
    snapshot_id: str

    @classmethod
    def from_env(cls, snapshot_id: str) -> "BigQueryCacheConfig":
        project_id = (
            os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or ""
        ).strip()
        dataset = (os.environ.get("BIGQUERY_DATASET") or DEFAULT_DATASET).strip()
        location = (os.environ.get("BIGQUERY_LOCATION") or DEFAULT_LOCATION).strip()
        if "." in dataset:
            dataset_project, dataset_id = dataset.split(".", 1)
            project_id = project_id or dataset_project
            dataset = dataset_id
        return cls(project_id=project_id, dataset=dataset, location=location, snapshot_id=snapshot_id)

    def table(self, table_id: str) -> str:
        if not self.project_id:
            raise BigQueryCacheError("Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT for BigQuery runtime mode")
        return f"`{self.project_id}.{self.dataset}.{table_id}`"


def selected_data_backend(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("COPILOT_DATA_BACKEND") or "local").strip().lower()


def fallback_to_local_enabled() -> bool:
    return (os.environ.get("COPILOT_BIGQUERY_FALLBACK_TO_LOCAL") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _client(cfg: BigQueryCacheConfig) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime deps
        raise BigQueryCacheError("Install google-cloud-bigquery for BigQuery runtime mode") from exc
    return bigquery.Client(project=cfg.project_id, location=cfg.location)


def _query_rows(cfg: BigQueryCacheConfig, table_id: str, columns: list[str]) -> list[dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    cols = ", ".join(f"`{c}`" for c in columns)
    sql = f"SELECT {cols} FROM {cfg.table(table_id)} WHERE snapshot_id = @snapshot_id"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("snapshot_id", "STRING", cfg.snapshot_id)]
    )
    rows = _client(cfg).query(sql, location=cfg.location, job_config=job_config).result()
    return [dict(row.items()) for row in rows]


def load_pricing_cache(snapshot_id: str) -> dict[str, float]:
    cfg = BigQueryCacheConfig.from_env(snapshot_id)
    rows = _query_rows(cfg, "pricing_cache", ["product_id", "predicted_price_change_pct"])
    return {str(r["product_id"]): float(r["predicted_price_change_pct"]) for r in rows}


def load_sentiment_cache(snapshot_id: str) -> dict[str, dict[str, float]]:
    cfg = BigQueryCacheConfig.from_env(snapshot_id)
    rows = _query_rows(cfg, "sentiment_cache", ["product_id", "n_reviews", "p_pos", "p_neu", "p_neg"])
    return {
        str(r["product_id"]): {
            "n_reviews": float(r.get("n_reviews") or 0.0),
            "p_pos": float(r.get("p_pos") or 0.0),
            "p_neu": float(r.get("p_neu") or 0.0),
            "p_neg": float(r.get("p_neg") or 0.0),
        }
        for r in rows
    }


def log_decision_run(
    *,
    snapshot_id: str,
    owner_id: str,
    run_id: str,
    variant: str,
    goal: str,
    ranked_actions: list[dict[str, Any]],
    chosen_product_id: str | None = None,
    confidence_rating: int | None = None,
) -> int:
    """Write ranked actions to operator_decision_log. Returns number of rows written."""
    from google.cloud import bigquery as bq  # type: ignore

    cfg = BigQueryCacheConfig.from_env(snapshot_id)
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "event_ts": now,
            "snapshot_id": snapshot_id,
            "owner_id": owner_id,
            "run_id": run_id,
            "variant": variant,
            "event": "decision_submitted",
            "goal": goal,
            "product_id": str(a.get("product_id") or ""),
            "accepted": str(a.get("product_id") or "") == chosen_product_id if chosen_product_id else True,
            "confidence_rating": confidence_rating,
            "metadata_json": json.dumps({k: v for k, v in a.items() if k != "product_id"}, default=str),
        }
        for a in ranked_actions
    ]
    if not rows:
        return 0
    job_config = bq.LoadJobConfig(
        source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bq.WriteDisposition.WRITE_APPEND,
    )
    _client(cfg).load_table_from_json(
        rows,
        f"{cfg.project_id}.{cfg.dataset}.operator_decision_log",
        location=cfg.location,
        job_config=job_config,
    ).result()
    return len(rows)


def load_inventory_cache(snapshot_id: str) -> dict[str, dict[str, object]]:
    cfg = BigQueryCacheConfig.from_env(snapshot_id)
    rows = _query_rows(
        cfg,
        "inventory_cache",
        [
            "product_id",
            "stock_status",
            "risk_flag",
            "on_hand_units",
            "safety_stock_units",
            "available_to_sell",
            "mean_daily_revenue",
            "total_units_sold",
            "total_returns",
        ],
    )
    return {
        str(r["product_id"]): {
            "stock_status": str(r.get("stock_status") or "unknown"),
            "risk_flag": bool(r.get("risk_flag")),
            "on_hand_units": float(r.get("on_hand_units") or 0.0),
            "safety_stock_units": float(r.get("safety_stock_units") or 0.0),
            "available_to_sell": float(r.get("available_to_sell") or 0.0),
            "mean_daily_revenue": float(r.get("mean_daily_revenue") or 0.0),
            "total_units_sold": float(r.get("total_units_sold") or 0.0),
            "total_returns": float(r.get("total_returns") or 0.0),
        }
        for r in rows
    }
