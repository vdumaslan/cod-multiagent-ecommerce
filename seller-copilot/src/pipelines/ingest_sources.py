from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
from datasets import load_dataset
from google.cloud import bigquery


def load_hf_sample(dataset_id: str, subset: str | None, split: str, max_rows: int) -> pd.DataFrame:
    if subset:
        ds = load_dataset(dataset_id, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if idx >= max_rows:
            break
        rows.append(dict(row))
    return pd.DataFrame(rows)


def upload_dataframe_to_bigquery(
    df: pd.DataFrame, project_id: str, dataset: str, table: str
) -> None:
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded rows={len(df)} -> {table_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", default="stg_amazon_reviews")
    parser.add_argument("--hf-dataset", default="McAuley-Lab/Amazon-Reviews-2023")
    parser.add_argument("--hf-subset", default="raw_review_All_Beauty")
    parser.add_argument("--split", default="full")
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", args.project_id)
    subset = args.hf_subset.strip() or None
    df = load_hf_sample(args.hf_dataset, subset, args.split, args.max_rows)
    if df.empty:
        raise RuntimeError("Ingestion returned zero rows.")

    upload_dataframe_to_bigquery(df, args.project_id, args.dataset, args.table)


if __name__ == "__main__":
    main()
