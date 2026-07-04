from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.feature_extraction.text import TfidfVectorizer

from runtime.inventory_agent import InventoryAgentConfig, attach_inventory_to_ranked_actions


@dataclass(frozen=True)
class OrchestratorConfig:
    snapshot_id: str
    artifacts_root: Path
    registry_path: Path | None = None

    retrieval_k: int = 10
    retrieval_candidate_k: int = 80
    dense_weight: float = 0.7
    lexical_weight: float = 0.3

    top_n_actions: int = 3
    candidate_pool: int = 40


@dataclass
class OrchestratorContext:
    registry: dict[str, Any]
    retriever: dict[str, Any]
    products: pd.DataFrame
    signals: pd.DataFrame
    inventory: pd.DataFrame
    sales_summary: pd.DataFrame | None = None
    product_owner: dict[str, str] | None = None

    # Demo improvements
    pricing_cache: dict[str, float] | None = None
    sentiment_cache: dict[str, dict[str, float]] | None = None
    demo_allowlist: set[str] | None = None


def _read_registry(registry_path: Path) -> dict[str, Any]:
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _norm01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _load_products(*, artifacts_root: Path, snapshot_id: str) -> pd.DataFrame:
    return pd.read_parquet(artifacts_root / "data_snapshots" / snapshot_id / "products.parquet")


def _load_product_signals(*, artifacts_root: Path, snapshot_id: str) -> pd.DataFrame:
    return pd.read_parquet(artifacts_root / "data_snapshots" / snapshot_id / "product_signals.parquet")


def _load_inventory(*, artifacts_root: Path, snapshot_id: str) -> pd.DataFrame:
    return pd.read_parquet(artifacts_root / "synthetic" / snapshot_id / "inventory_skus.parquet")


def _load_sales_summary(*, artifacts_root: Path, snapshot_id: str) -> pd.DataFrame:
    """
    Aggregate synthetic daily sales into per-product signals used by the orchestrator.

    Output columns:
    - product_id
    - mean_daily_revenue
    - total_returns
    """
    sales_path = artifacts_root / "synthetic" / snapshot_id / "sales_daily.parquet"
    if not sales_path.is_file():
        return pd.DataFrame({"product_id": [], "mean_daily_revenue": [], "total_returns": []})
    df = pd.read_parquet(sales_path, columns=["product_id", "gross_revenue_usd", "return_units"])
    g = df.groupby("product_id", as_index=False).agg(
        mean_daily_revenue=("gross_revenue_usd", "mean"),
        total_returns=("return_units", "sum"),
    )
    return g

def _apply_prefix(s: str, prefix: str | None) -> str:
    if not prefix:
        return s
    return f"{prefix}{s}"


def _load_sentence_transformer_model(model_name: str, *, device: str, max_seq_length: int) -> SentenceTransformer:
    try:
        model = SentenceTransformer(model_name, device=device)
    except TypeError as e:
        # Portable local snapshots may contain HF Transformer weights but miss
        # SentenceTransformers pooling metadata. E5 uses mean pooling, so build
        # the equivalent runtime model without requiring a metadata download.
        if "Pooling.__init__" not in str(e) or "e5" not in model_name.lower():
            raise
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    model.max_seq_length = max_seq_length
    return model


def _load_retriever_from_index_meta(index_meta_path: Path) -> dict[str, Any]:
    meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    paths = meta["paths"]
    base_dir = index_meta_path.parent

    def _resolve(p: str) -> str:
        # Support both absolute paths (legacy) and relative paths (portable artifacts).
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((base_dir / pp).resolve())

    model_name = str(meta["model_name"])
    # Build-time metadata can say cuda; default to CPU for portable demo/runtime.
    device = os.getenv("COPILOT_V2_RETRIEVER_DEVICE") or "cpu"
    model = _load_sentence_transformer_model(
        model_name,
        device=device,
        max_seq_length=int(meta.get("max_seq_length") or 384),
    )
    index = faiss.read_index(_resolve(str(paths["faiss_index"])))
    corpus = pd.read_parquet(_resolve(str(paths["corpus_parquet"])))
    docs = corpus[str(meta.get("doc_col", "product_document"))].astype(str).tolist()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(docs)
    return {"meta": meta, "model": model, "model_name": model_name, "index": index, "corpus": corpus, "tfidf": tfidf, "tfidf_matrix": tfidf_matrix}


def retrieve_evidence(
    query: str,
    retriever: dict[str, Any],
    *,
    k: int,
    candidate_k: int,
    dense_weight: float,
    lexical_weight: float,
) -> pd.DataFrame:
    meta = retriever["meta"]
    q = str(query)
    if bool(meta.get("use_prefixes", False)):
        q = _apply_prefix(q, str(meta.get("query_prefix") or ""))
    q_dense = retriever["model"].encode([q], normalize_embeddings=True)
    dense_scores, dense_idx = retriever["index"].search(q_dense.astype(np.float32), int(candidate_k))
    cand_idx = dense_idx[0]
    cand_dense = dense_scores[0].astype(np.float64)

    q_lex = retriever["tfidf"].transform([str(query)])
    lex_all = (retriever["tfidf_matrix"] @ q_lex.T).toarray().ravel()
    cand_lex = lex_all[cand_idx]

    dense_n = _norm01(cand_dense)
    lex_n = _norm01(cand_lex)
    hybrid = float(dense_weight) * dense_n + float(lexical_weight) * lex_n
    order = np.argsort(-hybrid)[: int(k)]

    corpus = retriever["corpus"].iloc[cand_idx[order]].copy().reset_index(drop=True)
    corpus["score"] = hybrid[order]
    corpus["score_dense"] = cand_dense[order]
    corpus["score_lexical"] = cand_lex[order]
    corpus["retrieval_model"] = str(retriever["model_name"])
    return corpus


def _default_owner_for_product(product_id: str, *, n_owners: int = 8) -> str:
    """
    Deterministically assign a synthetic owner/store id to a product_id.
    This is a stand-in for real multi-tenant data (seller/store dimension).
    """
    pid = str(product_id or "")
    h = zlib.crc32(pid.encode("utf-8")) & 0xFFFFFFFF
    return f"store_{h % int(n_owners):02d}"


def build_ranked_plans(
    *,
    cfg: OrchestratorConfig,
    goal: str,
    owner_id: str | None = None,
    constraints: dict[str, Any] | None = None,
    horizon_days: int = 7,
    enable_pricing: bool = True,
    enable_sentiment: bool = True,
    ctx: OrchestratorContext | None = None,
) -> dict[str, Any]:
    registry = ctx.registry if ctx is not None else (_read_registry(cfg.registry_path) if cfg.registry_path else {})
    index_meta = cfg.artifacts_root / "indexes" / cfg.snapshot_id / "dense" / "intfloat_e5-large-v2" / "index_meta.json"
    retriever = ctx.retriever if ctx is not None else _load_retriever_from_index_meta(index_meta)

    evidence = retrieve_evidence(
        goal,
        retriever,
        k=cfg.retrieval_k,
        candidate_k=cfg.retrieval_candidate_k,
        dense_weight=cfg.dense_weight,
        lexical_weight=cfg.lexical_weight,
    )

    products = ctx.products if ctx is not None else _load_products(artifacts_root=cfg.artifacts_root, snapshot_id=cfg.snapshot_id)
    signals = ctx.signals if ctx is not None else _load_product_signals(artifacts_root=cfg.artifacts_root, snapshot_id=cfg.snapshot_id)
    inventory = ctx.inventory if ctx is not None else _load_inventory(artifacts_root=cfg.artifacts_root, snapshot_id=cfg.snapshot_id)
    sales_summary = None
    if ctx is not None and getattr(ctx, "sales_summary", None) is not None:
        sales_summary = ctx.sales_summary
    product_owner = None
    if ctx is not None and getattr(ctx, "product_owner", None) is not None:
        product_owner = ctx.product_owner

    base_cols = [c for c in ["product_id", "title", "brand", "category", "subcategory", "price", "avg_rating"] if c in products.columns]
    if "product_id" not in base_cols:
        base_cols = ["product_id"] + base_cols

    top = evidence.merge(products[base_cols], on="product_id", how="left")
    if "product_id" in signals.columns:
        cols = [c for c in ["product_id", "positive_ratio", "recency_score"] if c in signals.columns]
        if cols:
            top = top.merge(signals[cols], on="product_id", how="left")
    if "product_id" in inventory.columns:
        cols = [
            c
            for c in [
                "product_id",
                "margin_pct",
                "available_to_sell",
                "mean_daily_revenue",
                "total_returns",
                "on_hand_units",
                "safety_stock_units",
            ]
            if c in inventory.columns
        ]
        if cols:
            top = top.merge(inventory[cols], on="product_id", how="left")

    if sales_summary is not None and "product_id" in sales_summary.columns:
        cols = [c for c in ["product_id", "mean_daily_revenue", "total_returns"] if c in sales_summary.columns]
        if cols:
            top = top.merge(sales_summary[cols], on="product_id", how="left")

    def _ensure_num(col: str) -> None:
        if col not in top.columns:
            top[col] = 0.0
        top[col] = pd.to_numeric(top[col], errors="coerce").fillna(0.0)

    for col in ["recency_score", "margin_pct", "available_to_sell", "positive_ratio", "mean_daily_revenue", "total_returns", "score"]:
        _ensure_num(col)

    # If the snapshot inventory provides on_hand/safety_stock but not available_to_sell,
    # derive a conservative sellable units estimate.
    if "available_to_sell" in top.columns and (top["available_to_sell"] == 0).all():
        if "on_hand_units" in top.columns and "safety_stock_units" in top.columns:
            on_hand = pd.to_numeric(top["on_hand_units"], errors="coerce").fillna(0.0)
            safety = pd.to_numeric(top["safety_stock_units"], errors="coerce").fillna(0.0)
            top["available_to_sell"] = (on_hand - safety).clip(lower=0.0)

    if ctx is not None and ctx.demo_allowlist:
        allow = set(ctx.demo_allowlist)
        top = top[top["product_id"].astype(str).isin(allow)].copy()

    if owner_id:
        oid = str(owner_id)
        if product_owner:
            mask = top["product_id"].astype(str).map(product_owner).fillna("") == oid
        else:
            mask = top["product_id"].astype(str).map(lambda x: _default_owner_for_product(x)) == oid
        top = top[mask].copy()

    by_score = top.sort_values(["score", "recency_score"], ascending=[False, False])
    # Candidate selection policy:
    # - Always return top_n_actions when possible.
    # - When enable_pricing=true and a pricing_cache is available, prefer SKUs present in the cache
    #   so UI never shows pricing.source="none". If we still can't fill the list, we allow a
    #   fallback pricing recommendation of 0.0% (pricing.source="fallback") for remaining slots.
    candidate_pids: list[str] = []
    all_pids_ranked = by_score["product_id"].astype(str).tolist()
    cache_pids: set[str] = set()
    if enable_pricing and ctx is not None and ctx.pricing_cache:
        cache_pids = set(str(x) for x in ctx.pricing_cache.keys())

    if cache_pids:
        for pid in all_pids_ranked:
            if pid in cache_pids:
                candidate_pids.append(pid)
                if len(candidate_pids) >= int(cfg.top_n_actions):
                    break

    # Fill remaining slots with next-best products (even if not priced) so we still return N actions.
    if len(candidate_pids) < int(cfg.top_n_actions):
        for pid in all_pids_ranked:
            if pid in candidate_pids:
                continue
            candidate_pids.append(pid)
            if len(candidate_pids) >= int(cfg.top_n_actions):
                break

    price_pred: dict[str, float] = {}
    if enable_pricing and ctx is not None and ctx.pricing_cache:
        price_pred = {pid: float(ctx.pricing_cache[pid]) for pid in candidate_pids if pid in ctx.pricing_cache}

    sent_pred: dict[str, dict[str, float]] = {}
    if enable_sentiment and ctx is not None and ctx.sentiment_cache:
        sent_pred = {pid: dict(ctx.sentiment_cache.get(pid) or {}) for pid in candidate_pids if pid in ctx.sentiment_cache}

    ranked_actions: list[dict[str, Any]] = []
    for pid in candidate_pids:
        row = by_score[by_score["product_id"].astype(str) == pid].head(1)
        r = row.iloc[0] if not row.empty else None
        s = sent_pred.get(pid) or {}
        evid_points: list[dict[str, Any]] = []
        if r is not None and "product_document" in r and isinstance(r.get("product_document"), str):
            txt = " ".join(str(r.get("product_document")).split())
            if len(txt) > 320:
                txt = txt[:317] + "..."
            evid_points.append({"type": "retrieval_doc", "text": txt, "score": float(r.get("score", 0.0))})
        if s:
            evid_points.append({"type": "sentiment_summary", **{k: float(s.get(k, 0.0)) for k in ["n_reviews", "p_pos", "p_neu", "p_neg"]}})

        ranked_actions.append(
            {
                "product_id": pid,
                "action_type": "reprice" if float(s.get("p_neg", 0.0)) <= 0.35 else "investigate",
                "recommended_price_change_pct": float(price_pred.get(pid, 0.0)),
                "pricing": {
                    "source": (
                        "cache"
                        if pid in price_pred
                        else ("fallback" if enable_pricing else "none")
                    )
                },
                "sentiment": s,
                "signals": {
                    "margin_pct": float(r.get("margin_pct", 0.0)) if r is not None else 0.0,
                    "available_to_sell": float(r.get("available_to_sell", 0.0)) if r is not None else 0.0,
                    "mean_daily_revenue": float(r.get("mean_daily_revenue", 0.0)) if r is not None else 0.0,
                    "total_returns": float(r.get("total_returns", 0.0)) if r is not None else 0.0,
                    "on_hand_units": float(r.get("on_hand_units", 0.0)) if r is not None else 0.0,
                    "safety_stock_units": float(r.get("safety_stock_units", 0.0)) if r is not None else 0.0,
                },
                "evidence": {"retrieval_score": float(r.get("score", 0.0)) if r is not None else 0.0, "points": evid_points[:3]},
                "horizon_days": int(horizon_days),
            }
        )

    # Inventory Agent (rule-based) attaches an inventory classification per candidate action.
    ranked_actions, inventory_by_pid = attach_inventory_to_ranked_actions(
        ranked_actions,
        cfg=InventoryAgentConfig(),
    )

    trace = {
        "snapshot_id": cfg.snapshot_id,
        "owner_id": str(owner_id) if owner_id else None,
        "retrieval_index_meta": str(index_meta),
        "pricing": {
            "wired": bool(enable_pricing),
            "source": (
                "cache+fallback"
                if (enable_pricing and ctx and ctx.pricing_cache)
                else ("none" if not enable_pricing else "fallback")
            ),
        },
        "sentiment": {"wired": bool(sent_pred), "source": "cache" if (ctx and ctx.sentiment_cache) else "none"},
        "demo_allowlist_size": len(ctx.demo_allowlist) if (ctx and ctx.demo_allowlist) else 0,
        "inventory": {"wired": bool(inventory_by_pid), "mode": "rules_v1"},
    }
    return {"input": {"goal": goal, "horizon_days": int(horizon_days), "constraints": constraints or {}}, "ranked_actions": ranked_actions, "trace": trace}

