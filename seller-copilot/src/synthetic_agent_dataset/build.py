"""
Build agent-ready Parquet tables matching Stage 1 schema.
Text pools are sampled from local McAuley Home & Kitchen JSONL when available.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _asin(rng: np.random.Generator) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "B" + "".join(rng.choice(list(alphabet), size=9).tolist())


def load_pools_from_raw(
    review_jsonl: Path | None,
    meta_jsonl: Path | None,
    *,
    max_review_lines: int = 200_000,
    max_meta_lines: int = 120_000,
) -> dict[str, list]:
    review_texts: list[str] = []
    review_titles: list[str] = []
    meta_titles: list[str] = []
    meta_descs: list[str] = []
    meta_brands: list[str] = []
    meta_cats: list[str] = []
    prices: list[float] = []

    if review_jsonl and review_jsonl.is_file():
        logger.info("Sampling review text pool from %s", review_jsonl)
        with review_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_review_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = o.get("text") or ""
                if isinstance(t, str) and len(t) > 35:
                    review_texts.append(t[:4000])
                rt = o.get("title") or ""
                if isinstance(rt, str) and len(rt) > 0:
                    review_titles.append(rt[:500])
        logger.info("Review pool size: %s", len(review_texts))
    else:
        logger.warning("No review JSONL — using fallback English templates.")

    if meta_jsonl and meta_jsonl.is_file():
        logger.info("Sampling meta pool from %s", meta_jsonl)
        with meta_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_meta_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ti = o.get("title") or ""
                if isinstance(ti, str) and len(ti) > 5:
                    meta_titles.append(ti[:500])
                desc = o.get("description") or ""
                if isinstance(desc, list):
                    desc = " ".join(str(x) for x in desc)
                if isinstance(desc, str) and len(desc) > 15:
                    meta_descs.append(desc[:8000])
                st = o.get("store") or o.get("brand") or ""
                if isinstance(st, str) and len(st) > 1:
                    meta_brands.append(st[:200])
                mc = o.get("main_category") or "Home_and_Kitchen"
                if isinstance(mc, str):
                    meta_cats.append(mc[:200])
                pr = o.get("price")
                if pr is not None:
                    ps = str(pr).replace("$", "").replace(",", "")
                    try:
                        prices.append(float(ps))
                    except ValueError:
                        pass
        logger.info("Meta pool: %s titles, %s prices", len(meta_titles), len(prices))
    return {
        "review_texts": review_texts,
        "review_titles": review_titles,
        "meta_titles": meta_titles,
        "meta_descs": meta_descs,
        "meta_brands": meta_brands,
        "meta_cats": meta_cats,
        "prices": prices,
    }


def _fallback_pools() -> dict[str, list]:
    return {
        "review_texts": [
            "Solid build quality for the price. Works as expected in our kitchen.",
            "Packaging was fine. Arrived on time. Would buy again for daily use.",
            "Easy to clean. No issues after several weeks of regular use.",
        ]
        * 400,
        "review_titles": ["Good value", "Nice product", "Works well", "Happy with purchase"] * 200,
        "meta_titles": [
            "Stainless Steel Kitchen Utensil Set",
            "Nonstick Cookware Pan 10 inch",
            "Glass Food Storage Containers with Lids",
        ]
        * 200,
        "meta_descs": [
            "Durable design for everyday cooking and meal prep at home.",
            "Designed for Home & Kitchen use with easy maintenance.",
        ]
        * 300,
        "meta_brands": ["HomeBrand", "KitchenPro", "CookRight", "DailyChef"] * 80,
        "meta_cats": ["Home_and_Kitchen"] * 500,
        "prices": list(np.clip(np.random.default_rng(1).lognormal(3.2, 0.9, 5000), 4.99, 899.0)),
    }


def build_agent_tables_from_pools(
    pools: dict[str, list],
    *,
    n_products: int,
    rng: np.random.Generator,
    min_reviews_per_product: int = 18,
    max_reviews_per_product: int = 120,
    recent_year_floor: int = 2018,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    fb = _fallback_pools()
    rt = pools["review_texts"] if len(pools["review_texts"]) > 100 else fb["review_texts"]
    rti = pools["review_titles"] if len(pools["review_titles"]) > 20 else fb["review_titles"]
    mt = pools["meta_titles"] if len(pools["meta_titles"]) > 20 else fb["meta_titles"]
    md = pools["meta_descs"] if len(pools["meta_descs"]) > 20 else fb["meta_descs"]
    mb = pools["meta_brands"] if len(pools["meta_brands"]) > 10 else fb["meta_brands"]
    mc = pools["meta_cats"] if len(pools["meta_cats"]) > 5 else fb["meta_cats"]
    price_pool = pools["prices"] if len(pools.get("prices", [])) > 50 else fb["prices"]

    pids: list[str] = []
    seen: set[str] = set()
    while len(pids) < n_products:
        a = _asin(rng)
        if a not in seen:
            seen.add(a)
            pids.append(a)

    nrevs = rng.integers(min_reviews_per_product, max_reviews_per_product + 1, size=n_products)

    t0 = datetime(recent_year_floor, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    span_ms = (t1 - t0).total_seconds() * 1000

    meta_rows: list[dict[str, Any]] = []
    for pi, pid in enumerate(pids):
        title = str(rng.choice(mt))
        desc = str(rng.choice(md))
        brand = str(rng.choice(mb))
        cat = str(rng.choice(mc)) if mc else "Home_and_Kitchen"
        price = float(rng.choice(price_pool))
        avg_meta = float(rng.uniform(3.8, 4.9))
        rnm = int(rng.integers(80, 8000))
        doc = " ".join((title + " " + desc).strip()[:20000].split())
        meta_rows.append(
            {
                "product_id": pid,
                "title": title[:500],
                "brand": brand[:200],
                "category": cat,
                "subcategory": cat,
                "description": desc[:15000],
                "price": price,
                "avg_rating": avg_meta,
                "rating_count": rnm,
                "product_document": doc,
                "_nrev": int(nrevs[pi]),
            }
        )

    # --- Vectorized review generation (was slow Python append loop) ---
    logger.info("Building %s review rows (vectorized)...", int(nrevs.sum()))
    rt_arr = np.asarray(rt, dtype=object)
    rti_arr = np.asarray(rti, dtype=object)
    n_total = int(nrevs.sum())
    pid_rep = np.repeat(np.asarray(pids, dtype=object), nrevs)

    idx1 = rng.integers(0, len(rt_arr), size=n_total)
    idx2 = rng.integers(0, len(rt_arr), size=n_total)
    mask_ex = rng.random(n_total) < 0.4
    s1 = pd.Series(rt_arr[idx1], dtype=object, copy=False)
    s2 = pd.Series(rt_arr[idx2], dtype=object, copy=False)
    texts = s1 + np.where(mask_ex, " " + s2, "")
    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip().str[:5000]
    short = texts.str.len() < 30
    if short.any():
        texts = texts.where(~short, (s1 + " " + s1.str[:250]).str[:2000])

    rv_titles = pd.Series(rti_arr[rng.integers(0, len(rti_arr), size=n_total)], dtype=object)
    ratings = rng.choice(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), size=n_total, p=[0.02, 0.04, 0.08, 0.28, 0.58])
    helpful = rng.poisson(3, size=n_total).astype(np.float64)
    t0_ms = t0.timestamp() * 1000
    ts_ms = t0_ms + rng.random(n_total) * span_ms
    event_ts = pd.to_datetime(ts_ms, unit="ms", utc=True)

    review_id = np.arange(n_total, dtype=np.int64)
    reviews = pd.DataFrame(
        {
            "review_id": "R" + pd.Series(review_id).astype(str),
            "product_id": pid_rep,
            "review_title": rv_titles.astype(str).str[:500],
            "review_text": texts,
            "rating": ratings,
            "helpful_vote": helpful,
            "event_ts": event_ts,
        }
    )

    meta_df = pd.DataFrame(meta_rows).drop(columns=["_nrev"], errors="ignore")
    g = reviews.groupby("product_id", as_index=False).agg(
        review_count=("review_id", "count"),
        avg_review_rating=("rating", "mean"),
        avg_helpful_vote=("helpful_vote", "mean"),
        last_review_ts=("event_ts", "max"),
    )
    products = meta_df.merge(g, on="product_id", how="inner")

    sig = (
        reviews.assign(pos=(reviews["rating"] >= 4).astype(int))
        .groupby("product_id", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            positive_ratio=("pos", "mean"),
            sentiment_score=("rating", "mean"),
            recency_score=("event_ts", lambda s: float(s.notna().mean())),
        )
    )

    retrieval = products[["product_id", "product_document", "category", "subcategory", "price", "avg_rating"]].copy()

    report = {
        "mode": "replicated_agent_baseline",
        "n_products": n_products,
        "n_reviews": len(reviews),
        "pools": {"review_text_n": len(rt), "meta_title_n": len(mt)},
    }
    return products, reviews, sig, retrieval, report


def write_agent_bundle(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    features: pd.DataFrame,
    retrieval: pd.DataFrame,
    *,
    agent_dir: Path,
    quality_report: dict,
    manifest_extra: dict[str, Any],
) -> None:
    agent_dir = Path(agent_dir).resolve()
    agent_dir.mkdir(parents=True, exist_ok=True)
    art = Path("seller-copilot/artifacts/stage1")
    art.mkdir(parents=True, exist_ok=True)

    products.to_parquet(agent_dir / "products.parquet", index=False)
    reviews.to_parquet(agent_dir / "reviews.parquet", index=False)
    features.to_parquet(agent_dir / "product_signals.parquet", index=False)
    retrieval.to_parquet(agent_dir / "retrieval_corpus.parquet", index=False)
    products.to_parquet(art / "products_curated.parquet", index=False)
    reviews.to_parquet(art / "reviews_curated.parquet", index=False)
    features.to_parquet(art / "product_signals_curated.parquet", index=False)

    row_counts = {
        "products": len(products),
        "reviews": len(reviews),
        "product_signals": len(features),
        "retrieval_corpus": len(retrieval),
    }
    manifest = {
        "schema_version": 1,
        "purpose": "Home & Kitchen agent tables for Seller Copilot (replicated from McAuley extract sampling).",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ingestion": manifest_extra.get("ingestion", {}),
        "quality_gates": manifest_extra.get("quality_gates", {}),
        "row_counts": row_counts,
        "files": {
            "products.parquet": "Product metadata and review aggregates.",
            "reviews.parquet": "Review text and ratings.",
            "product_signals.parquet": "Per-product signals.",
            "retrieval_corpus.parquet": "RAG bundle.",
        },
        "columns": {
            "products": list(products.columns),
            "reviews": list(reviews.columns),
            "product_signals": list(features.columns),
            "retrieval_corpus": list(retrieval.columns),
        },
        "build": quality_report,
    }
    (agent_dir / "agent_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "local_products": row_counts["products"],
        "local_reviews": row_counts["reviews"],
        "local_product_signals": row_counts["product_signals"],
    }
    payload = {"summary": summary, "quality_report": quality_report}
    (art / "curation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("seller-copilot/artifacts/quality_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
