"""
Local two-pass streaming over Amazon Reviews 2023 JSONL files.

Pass 1: scan all reviews, apply the same basic filters as Stage 1, count per product_id.
Pass 2: take top ``max_products`` by count, re-scan reviews + meta for those IDs only.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from data_acquisition.config import Stage1Config

logger = logging.getLogger(__name__)

# Progress: log every N lines while scanning large files
_PROGRESS_EVERY_LINES = 2_000_000


def _resolve_path(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _review_basic_ok(obj: dict[str, Any], cfg: Stage1Config) -> bool:
    """Mirror normalize_reviews + basic_ok in stage1_amazon_hk_flow (ingest_raw)."""
    pid = obj.get("parent_asin") or obj.get("asin") or obj.get("product_id")
    if not pid:
        return False
    text = obj.get("text") or obj.get("content") or ""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = " ".join(str(text).split()).strip()
    if len(text) < cfg.min_review_chars:
        return False
    try:
        rating = float(obj.get("rating"))
    except (TypeError, ValueError):
        return False
    if not (1 <= rating <= 5):
        return False
    ts = pd.to_datetime(obj.get("timestamp"), errors="coerce", unit="ms", utc=True)
    if pd.isna(ts):
        return False
    if int(ts.year) < cfg.recent_year_floor:
        return False
    return True


def pass1_count_reviews_per_product(reviews_path: Path, cfg: Stage1Config) -> Counter[str]:
    """Full-file scan: count qualifying reviews per product_id."""
    counts: Counter[str] = Counter()
    path = _resolve_path(reviews_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    logger.info("Pass 1: counting reviews per product from %s", path)
    n_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for n_lines, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _review_basic_ok(obj, cfg):
                continue
            pid = obj.get("parent_asin") or obj.get("asin") or obj.get("product_id")
            counts[str(pid)] += 1
            if n_lines % _PROGRESS_EVERY_LINES == 0:
                logger.info("Pass 1: processed %s lines, %s products with >=1 qualifying review", f"{n_lines:,}", f"{len(counts):,}")

    logger.info("Pass 1 done: %s lines read, %s distinct product_ids after basic filters", f"{n_lines:,}", f"{len(counts):,}")
    return counts


def select_top_product_ids(counts: Counter[str], max_products: int) -> set[str]:
    """Top ``max_products`` by review count (desc)."""
    top = counts.most_common(max_products)
    return {pid for pid, _ in top}


def pass2_stream_reviews_for_products(
    reviews_path: Path,
    product_ids: set[str],
    cfg: Stage1Config,
) -> pd.DataFrame:
    """Second pass: all review lines for selected product IDs (quality gates applied in ``curate``)."""
    path = _resolve_path(reviews_path)
    rows: list[dict[str, Any]] = []
    cap = cfg.max_reviews if cfg.max_reviews > 0 else None
    logger.info(
        "Pass 2a: collecting reviews for %s products (cap=%s)",
        f"{len(product_ids):,}",
        f"{cap:,}" if cap else "none",
    )

    n_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for n_lines, line in enumerate(f, 1):
            if cap is not None and len(rows) >= cap:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("parent_asin") or obj.get("asin") or obj.get("product_id")
            if not pid or str(pid) not in product_ids:
                continue
            rows.append(obj)
            if n_lines % _PROGRESS_EVERY_LINES == 0:
                logger.info("Pass 2a: scanned %s lines, collected %s reviews", f"{n_lines:,}", f"{len(rows):,}")

    logger.info("Pass 2a done: %s raw review rows for curation", f"{len(rows):,}")
    return pd.DataFrame(rows)


def _meta_pid(obj: dict[str, Any]) -> str | None:
    pid = obj.get("parent_asin") or obj.get("asin")
    return str(pid) if pid else None


def pass2_stream_meta_for_products(meta_path: Path, product_ids: set[str], cfg: Stage1Config) -> pd.DataFrame:
    """Scan meta JSONL; keep rows whose parent_asin is in ``product_ids``. Stops when all IDs found."""
    path = _resolve_path(meta_path)
    if not path.is_file():
        raise FileNotFoundError(f"Meta file not found: {path}")

    rows: list[dict[str, Any]] = []
    found: set[str] = set()
    cap = cfg.max_meta if cfg.max_meta > 0 else None
    target = len(product_ids)

    logger.info("Pass 2b: scanning meta for up to %s products", f"{target:,}")
    n_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for n_lines, line in enumerate(f, 1):
            if len(found) >= target:
                break
            if cap is not None and len(rows) >= cap:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = _meta_pid(obj)
            if not pid or pid not in product_ids:
                continue
            if pid in found:
                continue
            rows.append(obj)
            found.add(pid)
            if n_lines % _PROGRESS_EVERY_LINES == 0:
                logger.info(
                    "Pass 2b: scanned %s lines, matched %s / %s products",
                    f"{n_lines:,}",
                    f"{len(found):,}",
                    f"{target:,}",
                )

    logger.info("Pass 2b done: %s meta rows (matched %s distinct product_ids)", f"{len(rows):,}", f"{len(found):,}")
    return pd.DataFrame(rows)


def ingest_from_local_files(cfg: Stage1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-pass local ingestion using paths on ``cfg`` (``local_reviews_path``, ``local_meta_path``).
    """
    if not cfg.local_reviews_path or not cfg.local_meta_path:
        raise ValueError("local_reviews_path and local_meta_path must be set for local ingestion")

    reviews_path = _resolve_path(cfg.local_reviews_path)
    meta_path = _resolve_path(cfg.local_meta_path)

    counts = pass1_count_reviews_per_product(reviews_path, cfg)
    selected_ids = select_top_product_ids(counts, cfg.max_products)
    logger.info("Selected top %s product_ids by qualifying review count", f"{len(selected_ids):,}")

    raw_reviews = pass2_stream_reviews_for_products(reviews_path, selected_ids, cfg)
    raw_meta = pass2_stream_meta_for_products(meta_path, selected_ids, cfg)
    return raw_reviews, raw_meta
