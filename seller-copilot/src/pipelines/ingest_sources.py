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


def _as_flat_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    if value is None:
        return ""
    return str(value)


def _source_row_limit(source_cfg: dict[str, Any], default_rows: int, override_rows: int | None) -> int:
    if override_rows is not None:
        return int(override_rows)
    return int(source_cfg.get("target_rows", default_rows))


def _read_hf_stream_rows(
    dataset_id: str,
    split: str,
    max_rows: int,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    if subset:
        stream = load_dataset(dataset_id, subset, split=split, streaming=True)
    else:
        stream = load_dataset(dataset_id, split=split, streaming=True)

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(stream):
        if i >= max_rows:
            break
        rows.append(dict(row))
    return rows


def _read_hf_repo_jsonl_rows(repo_id: str, file_paths: list[str], max_rows: int) -> list[dict[str, Any]]:
    data_files = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}" for path in file_paths]
    stream = load_dataset("json", data_files=data_files, split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(stream):
        if i >= max_rows:
            break
        rows.append(dict(row))
    return rows


def _write_table_in_chunks(
    df: pd.DataFrame,
    project_id: str,
    dataset: str,
    table_name: str,
    chunk_size: int,
    partition_field: str,
    clustering_fields: list[str],
) -> None:
    if df.empty:
        raise RuntimeError(f"{table_name} has no rows to load.")
    for i, start in enumerate(range(0, len(df), chunk_size)):
        part = df.iloc[start : start + chunk_size].copy()
        disposition = "WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND"
        load_dataframe(
            part,
            project_id,
            dataset,
            table_name,
            partition_field=partition_field,
            clustering_fields=clustering_fields,
            write_disposition=disposition,
        )


def load_amazon_reviews(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    kind = cfg.get("kind")
    if kind == "huggingface_repo_jsonl":
        rows = _read_hf_repo_jsonl_rows(cfg["repo_id"], cfg["file_paths"], max_rows)
        if rows:
            frames.append(pd.DataFrame(rows))
        source_id = cfg["repo_id"]
    else:
        ds_id = cfg["dataset_id"]
        subsets = cfg.get("subsets") or [cfg.get("subset")]
        split = cfg.get("split", "full")
        if subsets and subsets[0] is not None:
            rows_per_subset = max(1, max_rows // max(1, len(subsets)))
            for subset in subsets:
                rows = _read_hf_stream_rows(ds_id, split=split, subset=subset, max_rows=rows_per_subset)
                if rows:
                    frames.append(pd.DataFrame(rows))
        else:
            rows = _read_hf_stream_rows(ds_id, split=split, max_rows=max_rows)
            if rows:
                frames.append(pd.DataFrame(rows))
        source_id = ds_id

    if not frames:
        raise RuntimeError("Amazon ingestion returned no rows.")
    raw = pd.concat(frames, ignore_index=True)

    fallback_product = (
        "p_"
        + pd.util.hash_pandas_object(_coalesce(raw, ["title", "content", "text"], ""), index=False)
        .astype("uint64")
        .astype(str)
    )
    mapped_rating = pd.to_numeric(_coalesce(raw, ["rating", "label"]), errors="coerce")
    mapped_rating = mapped_rating.where(mapped_rating.isin([1, 2, 3, 4, 5]), 1 + (mapped_rating * 4))
    mapped_rating = mapped_rating.clip(lower=1, upper=5)

    df = pd.DataFrame(
        {
            "product_id": _coalesce(raw, ["parent_asin", "asin", "product_id"], None).fillna(fallback_product).astype(str),
            "user_id": _coalesce(raw, ["user_id"], "unknown_user").astype(str),
            "rating": mapped_rating,
            "review_title": _coalesce(raw, ["title"], "").astype(str),
            "review_text": _coalesce(raw, ["text", "content"], "").astype(str),
            "event_ts": pd.to_numeric(_coalesce(raw, ["timestamp"]), errors="coerce"),
            "price": pd.to_numeric(_coalesce(raw, ["price"]), errors="coerce"),
            "currency": _coalesce(raw, ["currency"], "USD").astype(str),
            "source": source_id,
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["product_id"]).reset_index(drop=True)


def load_amazon_meta(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    if cfg.get("kind") != "huggingface_repo_jsonl":
        raise RuntimeError("amazon_meta currently supports huggingface_repo_jsonl only.")
    rows = _read_hf_repo_jsonl_rows(cfg["repo_id"], cfg["file_paths"], max_rows)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("Amazon meta ingestion returned no rows.")

    raw_price = _coalesce(raw, ["price"], "").apply(_as_flat_text)
    price_numeric = pd.to_numeric(raw_price.str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    currency = pd.Series(["USD" if "$" in p else "UNKNOWN" for p in raw_price], dtype="object")
    description = _coalesce(raw, ["description"], "").apply(_as_flat_text)
    features = _coalesce(raw, ["features"], "").apply(_as_flat_text)

    df = pd.DataFrame(
        {
            "product_id": _coalesce(raw, ["parent_asin"], "").astype(str),
            "title": _coalesce(raw, ["title"], "").astype(str),
            "description": (description + " " + features).str.strip(),
            "main_category": _coalesce(raw, ["main_category"], "").astype(str),
            "average_rating": pd.to_numeric(_coalesce(raw, ["average_rating"]), errors="coerce"),
            "rating_number": pd.to_numeric(_coalesce(raw, ["rating_number"]), errors="coerce"),
            "price": price_numeric,
            "currency": currency,
            "source": cfg["repo_id"],
            "ingested_at": _now_ts(),
        }
    )
    return df[df["product_id"].str.len() > 0].reset_index(drop=True)


def load_twitter_support(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    if cfg.get("kind") == "multi_huggingface_text":
        datasets = cfg.get("datasets", [])
        if not datasets:
            raise RuntimeError("multi_huggingface_text requires non-empty datasets list.")
        frames: list[pd.DataFrame] = []
        rows_per_ds = max(1, max_rows // len(datasets))
        for ds in datasets:
            rows = _read_hf_stream_rows(
                dataset_id=ds["dataset_id"],
                subset=ds.get("subset"),
                split=ds.get("split", "train"),
                max_rows=rows_per_ds,
            )
            f = pd.DataFrame(rows)
            if not f.empty:
                f["source_dataset"] = ds["dataset_id"]
                frames.append(f)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif cfg.get("kind") == "huggingface_text_classification":
        rows = _read_hf_stream_rows(
            dataset_id=cfg["dataset_id"],
            subset=cfg.get("subset"),
            split=cfg.get("split", "train"),
            max_rows=max_rows,
        )
        raw = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(cfg["url"], nrows=max_rows)

    if raw.empty:
        raise RuntimeError("Support ingestion returned no rows.")

    ticket_ids = pd.Series(range(len(raw))).astype(str).radd("tw_")
    text_series = _coalesce(raw, ["text", "content", "review_text"]).astype(str)
    df = pd.DataFrame(
        {
            "ticket_id": _coalesce(raw, ["tweet_id", "conversation_id", "id"], None).fillna(ticket_ids).astype(str),
            "user_id": _coalesce(raw, ["author_id", "user_id"], "unknown_user").astype(str),
            "text": text_series,
            "label": _coalesce(raw, ["inbound", "label"], "").astype(str),
            "created_at": _coalesce(raw, ["created_at", "timestamp"], "").astype(str),
            "source": _coalesce(raw, ["source_dataset"], cfg.get("dataset_id", "twitter_customer_support")).astype(str),
            "ingested_at": _now_ts(),
        }
    )
    return df.dropna(subset=["ticket_id"]).reset_index(drop=True)


def load_online_retail(cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    if cfg.get("kind") == "huggingface_tabular":
        rows = _read_hf_stream_rows(
            dataset_id=cfg["dataset_id"],
            subset=cfg.get("subset"),
            split=cfg.get("split", "train"),
            max_rows=max_rows,
        )
        raw = pd.DataFrame(rows)
    else:
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
    default_rows = int(pipe_cfg.get("max_rows_per_source", 100000))
    chunk_size = int(pipe_cfg.get("chunk_size", 20000))

    sources = cfg.get("sources", {})
    results: dict[str, int] = {}

    amazon_rows = _source_row_limit(sources["amazon_reviews"], default_rows, max_rows)
    amazon_df = load_amazon_reviews(sources["amazon_reviews"], amazon_rows)
    _write_table_in_chunks(
        amazon_df,
        project_id,
        dataset,
        sources["amazon_reviews"]["target_table"],
        chunk_size=chunk_size,
        partition_field="ingested_at",
        clustering_fields=["product_id", "source"],
    )
    results["stg_amazon_reviews"] = len(amazon_df)

    amazon_meta_rows = _source_row_limit(sources["amazon_meta"], default_rows, max_rows)
    amazon_meta_df = load_amazon_meta(sources["amazon_meta"], amazon_meta_rows)
    _write_table_in_chunks(
        amazon_meta_df,
        project_id,
        dataset,
        sources["amazon_meta"]["target_table"],
        chunk_size=chunk_size,
        partition_field="ingested_at",
        clustering_fields=["product_id", "source"],
    )
    results["stg_amazon_meta"] = len(amazon_meta_df)

    twitter_rows = _source_row_limit(sources["twitter_customer_support"], default_rows, max_rows)
    twitter_df = load_twitter_support(sources["twitter_customer_support"], twitter_rows)
    _write_table_in_chunks(
        twitter_df,
        project_id,
        dataset,
        sources["twitter_customer_support"]["target_table"],
        chunk_size=chunk_size,
        partition_field="ingested_at",
        clustering_fields=["ticket_id", "source"],
    )
    results["stg_twitter_support"] = len(twitter_df)

    retail_rows = _source_row_limit(sources["online_retail_ii"], default_rows, max_rows)
    retail_df = load_online_retail(sources["online_retail_ii"], retail_rows)
    _write_table_in_chunks(
        retail_df,
        project_id,
        dataset,
        sources["online_retail_ii"]["target_table"],
        chunk_size=chunk_size,
        partition_field="ingested_at",
        clustering_fields=["invoice_no", "source"],
    )
    results["stg_online_retail"] = len(retail_df)

    telco_rows = _source_row_limit(sources["telco_churn"], default_rows, max_rows)
    telco_df = load_telco_churn(sources["telco_churn"], telco_rows)
    _write_table_in_chunks(
        telco_df,
        project_id,
        dataset,
        sources["telco_churn"]["target_table"],
        chunk_size=chunk_size,
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
