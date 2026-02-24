from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from google.cloud import bigquery


def resolve_bq_settings(project_id: str | None, dataset: str | None) -> tuple[str, str]:
    p = project_id or os.getenv("GCP_PROJECT_ID", "")
    d = dataset or os.getenv("BIGQUERY_DATASET", "seller_copilot_prod")
    if not p:
        raise RuntimeError("Missing GCP project id. Set --project-id or GCP_PROJECT_ID.")
    return p, d


def load_bq_dataframe(project_id: str, sql: str) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    job = client.query(sql)
    return job.result().to_dataframe(create_bqstorage_client=False)


def split_by_hash(
    df: pd.DataFrame,
    key_cols: Iterable[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> pd.Series:
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise RuntimeError("Invalid split ratios.")
    keys = df[list(key_cols)].astype(str).agg("||".join, axis=1)
    # Stable deterministic split using pandas hash.
    h = pd.util.hash_pandas_object(keys, index=False).astype("uint64") % 10000
    train_cut = int(train_ratio * 10000)
    val_cut = int((train_ratio + val_ratio) * 10000)
    split = np.where(h < train_cut, "train", np.where(h < val_cut, "val", "test"))
    return pd.Series(split, index=df.index)


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
