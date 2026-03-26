from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests

_RERANKER = None  # lazy: sentence_transformers.CrossEncoder


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def _extract_keywords(goal: str) -> list[str]:
    tokens = [t.strip(" ,.;:!?()[]{}\"'").lower() for t in goal.split()]
    tokens = [t for t in tokens if len(t) >= 4]
    # Keep stable, deterministic top unique tokens.
    out: list[str] = []
    for t in tokens:
        if t not in out:
            out.append(t)
    return out[:6]


def _fetch_google_news_rss(query: str, limit: int) -> list[dict[str, Any]]:
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        out.append({"title": title, "url": link, "published_at": pub})
        if len(out) >= limit:
            break
    return out


def fetch_market_news(goal: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Free internet signal via Google News RSS search.
    No API key, lightweight, and works as optional context.
    """
    # Primary: goal + commerce terms; fallback: broad retail query if empty (rate limits / blocks).
    primary = f"{goal} retail ecommerce inventory pricing"
    out = _fetch_google_news_rss(primary, limit)
    if not out:
        out = _fetch_google_news_rss("retail ecommerce profit margin inventory", limit)
    return out[:limit]


def rerank_market_news(
    goal: str,
    news_items: list[dict[str, Any]],
    *,
    top_n: int = 10,
    use_model: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not news_items:
        return [], {"enabled": use_model, "model_name": None, "n_candidates": 0}
    if not use_model:
        return news_items[:top_n], {"enabled": False, "model_name": None, "n_candidates": len(news_items)}

    reranker = _get_reranker()
    pairs = [(goal, str(n.get("title", ""))) for n in news_items]
    scores = reranker.predict(pairs)
    scored = []
    for n, s in zip(news_items, scores):
        row = dict(n)
        row["relevance_score"] = float(s)
        scored.append(row)
    scored.sort(key=lambda x: float(x.get("relevance_score", 0.0)), reverse=True)
    return scored[:top_n], {
        "enabled": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "n_candidates": len(news_items),
    }


def summarize_market_signal(goal: str, news_items: list[dict[str, Any]]) -> dict[str, Any]:
    kws = _extract_keywords(goal)
    now = datetime.now(timezone.utc)
    total = len(news_items)
    mention_hits = 0
    freshness_scores: list[float] = []

    for n in news_items:
        t = str(n.get("title", "")).lower()
        if any(k in t for k in kws):
            mention_hits += 1
        pub_raw = n.get("published_at")
        try:
            dt = parsedate_to_datetime(pub_raw) if pub_raw else None
            if dt is not None and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt is not None:
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                freshness_scores.append(1.0 / (1.0 + age_days))
        except Exception:
            continue

    mention_ratio = (mention_hits / total) if total else 0.0
    freshness = (sum(freshness_scores) / len(freshness_scores)) if freshness_scores else 0.0
    trend_score = 0.65 * mention_ratio + 0.35 * freshness
    return {
        "keyword_hits": mention_hits,
        "total_headlines": total,
        "mention_ratio": mention_ratio,
        "freshness_score": freshness,
        "trend_score": trend_score,
        "keywords": kws,
        "sample_headlines": [n.get("title", "") for n in news_items[:5]],
    }


def market_inventory_suggestions(
    top: pd.DataFrame,
    market_signal: dict[str, Any],
    *,
    max_skus: int = 6,
) -> list[str]:
    """
    Inventory-aware SKU shortlist from existing evidence, biased by market trend.
    """
    if top.empty:
        return []
    t = top.copy()
    if "available_to_sell" in t.columns:
        t["available_to_sell"] = pd.to_numeric(t["available_to_sell"], errors="coerce").fillna(0.0)
    else:
        t["available_to_sell"] = 0.0
    if "recency_score" in t.columns:
        t["recency_score"] = pd.to_numeric(t["recency_score"], errors="coerce").fillna(0.0)
    else:
        t["recency_score"] = 0.0
    if "margin_pct" in t.columns:
        t["margin_pct"] = pd.to_numeric(t["margin_pct"], errors="coerce").fillna(0.0)
    else:
        t["margin_pct"] = 0.0

    trend_score = float(market_signal.get("trend_score", 0.0))
    # When trend is strong, prioritize demand/recency more. Otherwise be conservative on margin/inventory.
    w_rec = 0.55 + 0.20 * trend_score
    w_mar = 0.25 - 0.10 * trend_score
    w_inv = 0.20

    t["market_rank_score"] = (
        w_rec * t["recency_score"]
        + w_mar * t["margin_pct"]
        + w_inv * (t["available_to_sell"] > 0).astype(float)
    )
    return t.sort_values("market_rank_score", ascending=False)["product_id"].head(max_skus).tolist()
