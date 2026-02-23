from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bq_utils import load_dataframe
from pipeline_config import load_pipeline_config


def _now_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = None) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df))


def load_amazon_reviews(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    ds_id = cfg["dataset_id"]
    subsets = cfg.get("subsets", [])
    split = cfg.get("split", "full")
    frames: list[pd.DataFrame] = []
    rows_per_subset = max(1, max_rows // max(1, len(subsets)))

    for subset in subsets:
        stream = load_dataset(ds_id, subset, split=split, streaming=True)
        rows = []
        for i, row in enumerate(stream):
            if i >= rows_per_subset:
                break
            rows.append(dict(row))
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        raise RuntimeError("Amazon ingestion returned no rows.")
    raw = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame(
        {
            "product_id": _coalesce(raw, ["parent_asin", "asin"]).astype(str),
            "user_id": _coalesce(raw, ["user_id"]).astype(str),
            "rating": pd.to_numeric(_coalesce(raw, ["rating"]), errors="coerce"),
            "review_title": _coalesce(raw, ["title"], "").astype(str),
            "review_text": _coalesce(raw, ["text"], "").astype(str),
            "event_ts": pd.to_numeric(_coalesce(raw, ["timestamp"]), errors="coerce"),
            "price": pd.to_numeric(_coalesce(raw, ["price"]), errors="coerce"),
            "currency": _coalesce(raw, ["currency"], "USD").astype(str),
            "source": "amazon_reviews_2023",
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["product_id"]).reset_index(drop=True)


def load_twitter_support(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    raw = pd.read_csv(cfg["url"], nrows=max_rows)
    df = pd.DataFrame(
        {
            "ticket_id": _coalesce(raw, ["tweet_id", "conversation_id", "id"]).astype(str),
            "user_id": _coalesce(raw, ["author_id", "user_id"]).astype(str),
            "text": _coalesce(raw, ["text"]).astype(str),
            "label": _coalesce(raw, ["inbound", "label"], "").astype(str),
            "created_at": _coalesce(raw, ["created_at", "timestamp"], "").astype(str),
            "source": "twitter_customer_support",
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["ticket_id"]).reset_index(drop=True)


def load_online_retail(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    raw = pd.read_excel(cfg["url"], sheet_name=cfg.get("sheet_name", 0), nrows=max_rows)
    df = pd.DataFrame(
        {
            "invoice_no": _coalesce(raw, ["Invoice", "InvoiceNo"]).astype(str),
            "stock_code": _coalesce(raw, ["StockCode"]).astype(str),
            "description": _coalesce(raw, ["Description"], "").astype(str),
            "quantity": pd.to_numeric(_coalesce(raw, ["Quantity"]), errors="coerce"),
            "invoice_ts": _coalesce(raw, ["InvoiceDate"], "").astype(str),
            "unit_price": pd.to_numeric(_coalesce(raw, ["Price", "UnitPrice"]), errors="coerce"),
            "customer_id": _coalesce(raw, ["Customer ID", "CustomerID"]).astype(str),
            "country": _coalesce(raw, ["Country"], "").astype(str),
            "source": "online_retail_ii",
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["invoice_no"]).reset_index(drop=True)


def load_telco_churn(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    raw = pd.read_csv(cfg["url"], nrows=max_rows)
    df = pd.DataFrame(
        {
            "customer_id": _coalesce(raw, ["customerID"]).astype(str),
            "gender": _coalesce(raw, ["gender"], "").astype(str),
            "senior_citizen": pd.to_numeric(_coalesce(raw, ["SeniorCitizen"]), errors="coerce"),
            "tenure": pd.to_numeric(_coalesce(raw, ["tenure"]), errors="coerce"),
            "monthly_charges": pd.to_numeric(_coalesce(raw, ["MonthlyCharges"]), errors="coerce"),
            "total_charges": pd.to_numeric(_coalesce(raw, ["TotalCharges"]), errors="coerce"),
            "churn_label": _coalesce(raw, ["Churn"], "").astype(str),
            "source": "telco_churn",
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["customer_id"]).reset_index(drop=True)


def ingest_all(config_path: str, project_id: str, dataset: str, max_rows: int | None = None) -> dict[str, int]:
    cfg = load_pipeline_config(config_path)
    pipe_cfg = cfg.get("pipeline", {})
    max_rows_cfg = int(pipe_cfg.get("max_rows_per_source", 100000))
    rows = max_rows or max_rows_cfg

    sources = cfg.get("sources", {})
    results: dict[str, int] = {}

    amazon_df = load_amazon_reviews(sources["amazon_reviews"], rows)
    load_dataframe(
        amazon_df,
        project_id,
        dataset,
        sources["amazon_reviews"]["target_table"],
        partition_field="ingested_at",
        clustering_fields=["product_id", "source"],
    )
    results["stg_amazon_reviews"] = len(amazon_df)

    twitter_df = load_twitter_support(sources["twitter_customer_support"], rows)
    load_dataframe(
        twitter_df,
        project_id,
        dataset,
        sources["twitter_customer_support"]["target_table"],
        partition_field="ingested_at",
        clustering_fields=["ticket_id", "source"],
    )
    results["stg_twitter_support"] = len(twitter_df)

    retail_df = load_online_retail(sources["online_retail_ii"], rows)
    load_dataframe(
        retail_df,
        project_id,
        dataset,
        sources["online_retail_ii"]["target_table"],
        partition_field="ingested_at",
        clustering_fields=["invoice_no", "source"],
    )
    results["stg_online_retail"] = len(retail_df)

    telco_df = load_telco_churn(sources["telco_churn"], rows)
    load_dataframe(
        telco_df,
        project_id,
        dataset,
        sources["telco_churn"]["target_table"],
        partition_field="ingested_at",
        clustering_fields=["customer_id", "source"],
    )
    results["stg_telco_churn"] = len(telco_df)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="seller-copilot/config/pipeline.yaml")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out = ingest_all(args.config, args.project_id, args.dataset, args.max_rows)
    print(out)


if __name__ == "__main__":
    main()
