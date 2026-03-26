#!/usr/bin/env python3
"""
Load local agent_dataset + synthetic Parquet files into BigQuery.

Use after Stage 1 or generate_data.py + validate. Requires GCP credentials.

Usage:
  python seller-copilot/scripts/upload_to_bigquery.py --dry-run
  python seller-copilot/scripts/upload_to_bigquery.py

Env (see docs/GCP_SETUP.md):
  GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_LOCATION

Optional:
  --reset  drop target tables before load (agent + synthetic)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from common.bq import drop_tables, ensure_dataset, load_df  # noqa: E402


REPO = SCRIPT_DIR.parent

AGENT_TABLES = [
    ("products.parquet", "products", "updated_at", ["category", "subcategory", "product_id"]),
    ("reviews.parquet", "reviews", "ingested_at", ["product_id"]),
    ("product_signals.parquet", "product_signals", "updated_at", ["product_id"]),
    ("retrieval_corpus.parquet", "retrieval_corpus", "updated_at", ["category", "subcategory", "product_id"]),
]

SYNTHETIC_TABLES = [
    ("inventory_skus.parquet", "inventory_skus", None, ["product_id"]),
    ("suppliers.parquet", "suppliers", None, ["supplier_id"]),
    ("product_supplier_map.parquet", "product_supplier_map", None, ["product_id", "supplier_id"]),
    ("sales_daily.parquet", "sales_daily", "sale_date", ["product_id"]),
    ("marketing_spend_daily.parquet", "marketing_spend_daily", "date", ["product_id"]),
    ("store_kpis_weekly.parquet", "store_kpis_weekly", None, None),
]


def _bootstrap_env() -> None:
    load_dotenv(REPO / ".env")
    raw = os.getenv("GCP_SA_KEY_JSON", "").strip()
    if raw and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        d = REPO / ".secrets"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "gcp-sa-from-env.json"
        p.write_text(raw, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p.resolve())


def _prepare_agent_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    now = datetime.now(timezone.utc)
    if name == "products.parquet":
        df["updated_at"] = now
    elif name == "reviews.parquet":
        if "event_ts" in df.columns:
            df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
        df["ingested_at"] = now
    elif name == "product_signals.parquet":
        df["updated_at"] = now
    elif name == "retrieval_corpus.parquet":
        df["updated_at"] = now
    return df


def _prepare_synthetic_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if name == "sales_daily.parquet":
        df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.tz_localize(None)
    elif name == "marketing_spend_daily.parquet":
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    elif name == "store_kpis_weekly.parquet" and "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def main() -> None:
    _bootstrap_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, default=REPO / "data" / "agent_dataset")
    parser.add_argument("--synthetic-dir", type=Path, default=REPO / "data" / "synthetic")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", ""))
    parser.add_argument("--dataset", default=os.getenv("BIGQUERY_DATASET", "seller_copilot_prod"))
    parser.add_argument("--location", default=os.getenv("BIGQUERY_LOCATION", "US"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Drop tables before load")
    args = parser.parse_args()

    project = args.project.strip()
    if not project and not args.dry_run:
        print("Set GCP_PROJECT_ID or pass --project", file=sys.stderr)
        sys.exit(1)

    agent_dir = args.agent_dir.resolve()
    syn_dir = args.synthetic_dir.resolve()

    to_drop = [t for _, t, _, _ in AGENT_TABLES] + [t for _, t, _, _ in SYNTHETIC_TABLES]

    if args.dry_run:
        print("Dry run — would load:")
        for fn, tn, *_ in AGENT_TABLES:
            p = agent_dir / fn
            print(f"  {p} -> {project}.{args.dataset}.{tn}" if project else f"  {p} -> {tn}")
        for fn, tn, *_ in SYNTHETIC_TABLES:
            p = syn_dir / fn
            print(f"  {p} -> {project}.{args.dataset}.{tn}" if project else f"  {p} -> {tn}")
        return

    ensure_dataset(project, args.dataset, args.location)
    if args.reset:
        drop_tables(project, args.dataset, to_drop)

    for fn, table, part, cluster in AGENT_TABLES:
        path = agent_dir / fn
        if not path.is_file():
            print(f"Skip missing {path}", file=sys.stderr)
            continue
        df = _prepare_agent_df(fn, pd.read_parquet(path))
        load_df(project, args.dataset, table, df, partition_field=part, clustering_fields=cluster)
        print(f"Loaded {table}: {len(df)} rows")

    for fn, table, part, cluster in SYNTHETIC_TABLES:
        path = syn_dir / fn
        if not path.is_file():
            print(f"Skip missing {path}", file=sys.stderr)
            continue
        df = _prepare_synthetic_df(fn, pd.read_parquet(path))
        load_df(project, args.dataset, table, df, partition_field=part, clustering_fields=cluster)
        print(f"Loaded {table}: {len(df)} rows")

    print(f"Done: {project}.{args.dataset}.*")


if __name__ == "__main__":
    main()
