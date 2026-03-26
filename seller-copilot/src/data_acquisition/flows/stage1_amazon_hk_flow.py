from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from prefect import flow, task

from common.bq import drop_tables, ensure_dataset, load_df
from data_acquisition.config import Stage1Config, load_config
from data_acquisition.local_jsonl import ingest_from_local_files
from data_acquisition.quality import curate_products_and_reviews, normalize_meta, normalize_reviews


def _effective_max_reviews_remote(cfg: Stage1Config) -> int:
    """HTTP ingest: avoid unbounded download when max_reviews is 0 / unset."""
    if cfg.max_reviews > 0:
        return cfg.max_reviews
    return 800_000


def _effective_max_meta_remote(cfg: Stage1Config) -> int:
    if cfg.max_meta > 0:
        return cfg.max_meta
    return 200_000


def _read_jsonl_stream(repo_id: str, path: str, max_rows: int) -> list[dict]:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
    out: list[dict] = []
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if len(out) >= max_rows:
                break
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_jsonl_stream_filtered_by_product_ids(
    repo_id: str,
    path: str,
    product_ids: set[str],
    max_rows: int,
    *,
    stop_when_unique_products: int | None = None,
) -> list[dict]:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
    out: list[dict] = []
    seen_products: set[str] = set()
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if len(out) >= max_rows:
                break
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("parent_asin") or obj.get("asin") or obj.get("product_id")
            if not pid:
                continue
            pid = str(pid)
            if pid in product_ids:
                out.append(obj)
                seen_products.add(pid)
                if stop_when_unique_products is not None and len(seen_products) >= stop_when_unique_products:
                    break
    return out


@task
def ingest_raw(cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.use_local_files:
        if not cfg.local_reviews_path or not cfg.local_meta_path:
            raise ValueError("use_local_files=true requires local_reviews_path and local_meta_path in config (or env).")
        return ingest_from_local_files(cfg)

    max_rev = _effective_max_reviews_remote(cfg)
    max_meta = _effective_max_meta_remote(cfg)
    review_rows = _read_jsonl_stream(cfg.repo_id, cfg.review_path, max_rev)
    raw_reviews = pd.DataFrame(review_rows)

    # Choose high-density product_ids from valid reviews first, then fetch matching meta.
    reviews_norm = normalize_reviews(raw_reviews)
    basic_ok = (
        reviews_norm["product_id"].astype(str).str.len().gt(0)
        & reviews_norm["review_text"].astype(str).str.len().ge(cfg.min_review_chars)
        & reviews_norm["rating"].between(1, 5, inclusive="both")
        & (reviews_norm["event_ts"].dt.year.fillna(0).ge(cfg.recent_year_floor))
    )
    review_counts = (
        reviews_norm.loc[basic_ok]
        .groupby("product_id")
        .size()
        .sort_values(ascending=False)
    )
    selected_ids = review_counts.head(cfg.max_products).index.astype(str).tolist()
    product_ids_set = set(selected_ids)

    # Scan meta and keep only products observed in reviews.
    meta_rows = _read_jsonl_stream_filtered_by_product_ids(
        cfg.repo_id,
        cfg.meta_path,
        product_ids_set,
        max_meta,
        stop_when_unique_products=min(len(product_ids_set), max_meta),
    )
    raw_meta = pd.DataFrame(meta_rows)
    # Keep only reviews for selected product ids to improve overlap and downstream memory.
    pid_series = raw_reviews.get("parent_asin", raw_reviews.get("asin", pd.Series(dtype="object"))).astype(str)
    raw_reviews = raw_reviews[pid_series.isin(product_ids_set)]
    return raw_reviews, raw_meta


@task
def curate(
    raw_reviews: pd.DataFrame,
    raw_meta: pd.DataFrame,
    cfg: Stage1Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    products = normalize_meta(raw_meta)
    reviews = normalize_reviews(raw_reviews)
    products, reviews, features, report = curate_products_and_reviews(
        products,
        reviews,
        min_reviews_per_product=cfg.min_reviews_per_product,
        min_title_chars=cfg.min_title_chars,
        min_review_chars=cfg.min_review_chars,
        min_price_percentile=cfg.min_price_percentile,
        max_price_percentile=cfg.max_price_percentile,
        max_products=cfg.max_products,
        recent_year_floor=cfg.recent_year_floor,
    )
    return products, reviews, features, report


def _agent_dataset_manifest(
    cfg: Stage1Config,
    *,
    row_counts: dict[str, int],
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    features: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> dict:
    return {
        "schema_version": 1,
        "purpose": "Filtered Amazon Home & Kitchen subset for Seller Copilot agents, RAG, and tabular models (local export).",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ingestion": {
            "use_local_files": cfg.use_local_files,
            "local_reviews_path": cfg.local_reviews_path,
            "local_meta_path": cfg.local_meta_path,
        },
        "quality_gates": {
            "min_reviews_per_product": cfg.min_reviews_per_product,
            "min_title_chars": cfg.min_title_chars,
            "min_review_chars": cfg.min_review_chars,
            "min_price_percentile": cfg.min_price_percentile,
            "max_price_percentile": cfg.max_price_percentile,
            "max_products_cap": cfg.max_products,
            "recent_year_floor": cfg.recent_year_floor,
        },
        "row_counts": row_counts,
        "files": {
            "products.parquet": "Canonical product rows (metadata + merged review stats).",
            "reviews.parquet": "All curated reviews for those products (training / NLP / sentiment).",
            "product_signals.parquet": "Per-product aggregates (counts, sentiment, recency).",
            "retrieval_corpus.parquet": "Lean text bundle for RAG / embeddings (product_document + ids).",
        },
        "columns": {
            "products": list(products.columns),
            "reviews": list(reviews.columns),
            "product_signals": list(features.columns),
            "retrieval_corpus": list(retrieval.columns),
        },
    }


@task
def persist_local(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    features: pd.DataFrame,
    quality_report: dict,
    cfg: Stage1Config,
) -> dict[str, int]:
    out = Path("seller-copilot/artifacts/stage1")
    out.mkdir(parents=True, exist_ok=True)
    products.to_parquet(out / "products_curated.parquet", index=False)
    reviews.to_parquet(out / "reviews_curated.parquet", index=False)
    features.to_parquet(out / "product_signals_curated.parquet", index=False)
    summary = {
        "local_products": len(products),
        "local_reviews": len(reviews),
        "local_product_signals": len(features),
    }
    payload = {"summary": summary, "quality_report": quality_report}

    (out / "curation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (Path("seller-copilot/artifacts") / "quality_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Primary handoff for agents / models (no cloud required).
    agent_dir = Path(cfg.agent_dataset_dir)
    if not agent_dir.is_absolute():
        agent_dir = Path.cwd() / agent_dir
    agent_dir = agent_dir.resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)
    retrieval = products[["product_id", "product_document", "category", "subcategory", "price", "avg_rating"]].copy()
    products.to_parquet(agent_dir / "products.parquet", index=False)
    reviews.to_parquet(agent_dir / "reviews.parquet", index=False)
    features.to_parquet(agent_dir / "product_signals.parquet", index=False)
    retrieval.to_parquet(agent_dir / "retrieval_corpus.parquet", index=False)
    row_counts = {
        "products": len(products),
        "reviews": len(reviews),
        "product_signals": len(features),
        "retrieval_corpus": len(retrieval),
    }
    manifest = _agent_dataset_manifest(
        cfg,
        row_counts=row_counts,
        products=products,
        reviews=reviews,
        features=features,
        retrieval=retrieval,
    )
    (agent_dir / "agent_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary["agent_dataset_dir"] = str(agent_dir)
    summary["agent_row_counts"] = row_counts
    return summary


@task
def load_bigquery(products: pd.DataFrame, reviews: pd.DataFrame, features: pd.DataFrame, cfg: Stage1Config) -> dict[str, int]:
    if cfg.service_account_key_path and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.service_account_key_path
    ensure_dataset(cfg.project_id, cfg.dataset, cfg.location)
    tables = ["stg_home_products_raw", "stg_home_reviews_raw", "products", "reviews", "product_signals", "retrieval_corpus"]
    if cfg.reset_bigquery_tables:
        drop_tables(cfg.project_id, cfg.dataset, tables)

    now = datetime.now(timezone.utc)
    products = products.copy()
    products["updated_at"] = now
    reviews = reviews.copy()
    reviews["ingested_at"] = now
    features = features.copy()
    features["updated_at"] = now

    retrieval = products[["product_id", "product_document", "category", "subcategory", "price", "avg_rating"]].copy()
    retrieval["updated_at"] = now

    load_df(cfg.project_id, cfg.dataset, "products", products, partition_field="updated_at", clustering_fields=["category", "subcategory", "product_id"])
    load_df(cfg.project_id, cfg.dataset, "reviews", reviews, partition_field="ingested_at", clustering_fields=["product_id"])
    load_df(cfg.project_id, cfg.dataset, "product_signals", features, partition_field="updated_at", clustering_fields=["product_id"])
    load_df(cfg.project_id, cfg.dataset, "retrieval_corpus", retrieval, partition_field="updated_at", clustering_fields=["category", "subcategory", "product_id"])
    return {
        "products": len(products),
        "reviews": len(reviews),
        "product_signals": len(features),
        "retrieval_corpus": len(retrieval),
    }


@flow(name="stage1_amazon_home_kitchen")
def run_stage1(config_path: str = "seller-copilot/config/stage1_amazon_hk.yaml") -> dict[str, int]:
    cfg = load_config(config_path)
    raw_reviews, raw_meta = ingest_raw(cfg)
    products, reviews, features, report = curate(raw_reviews, raw_meta, cfg)
    out = persist_local(products, reviews, features, report, cfg)
    if cfg.require_bigquery:
        bq_out = load_bigquery(products, reviews, features, cfg)
        out.update(bq_out)
    return out

