from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_st() -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing sentence-transformers. Install it in `.venv-copilot-v2`.") from e
    return {"SentenceTransformer": SentenceTransformer}


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / (ranks + 1)))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    # Binary relevance for self-match at rank 0.
    dcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
    return float(np.mean(dcg))


@dataclass(frozen=True)
class DenseTrial:
    model: str
    max_seq_length: int
    use_prefixes: bool
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    n_evaluated: int
    phase: str  # screening | refine


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _encode_docs(
    *,
    model_name: str,
    docs: list[str],
    max_seq_length: int,
    use_prefixes: bool,
    batch_size: int,
    device: str,
) -> np.ndarray:
    st = _require_st()
    SentenceTransformer = st["SentenceTransformer"]
    mdl = SentenceTransformer(model_name, device=device if device != "auto" else None)
    try:
        mdl.max_seq_length = int(max_seq_length)
    except Exception:
        pass

    if use_prefixes:
        q_prefix = "query: "
        p_prefix = "passage: "
    else:
        q_prefix = ""
        p_prefix = ""

    # Use passage prefix for documents. Query prefix used when encoding queries (titles).
    doc_texts = [p_prefix + str(d) for d in docs]
    doc_emb = mdl.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return doc_emb, mdl, q_prefix


def _eval_self_match_ranks(*, doc_emb: np.ndarray, q_emb: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # Rank of self-match for each query index, using cosine similarity (dot product on normalized vectors).
    ranks: list[int] = []
    for qi in indices:
        sims = (doc_emb @ q_emb[qi]).astype(np.float64)
        order = np.argsort(-sims)
        ranks.append(int(np.where(order == qi)[0][0]))
    return np.asarray(ranks, dtype=np.int64)


def _pick_best(trials: Iterable[DenseTrial]) -> DenseTrial:
    # Lexicographic: recall@k, then ndcg@k, then mrr.
    best = None
    for t in trials:
        if best is None:
            best = t
            continue
        if (t.recall_at_k, t.ndcg_at_k, t.mrr) > (best.recall_at_k, best.ndcg_at_k, best.mrr):
            best = t
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--artifacts-root", default="copilot-v2/artifacts")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--screening-n", type=int, default=3000)
    p.add_argument("--full-n", type=int, default=7500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5"],
        help="One or more dense encoder names to tune.",
    )
    p.add_argument("--max-seq-lengths", nargs="+", type=int, default=[384, 512])
    p.add_argument("--use-prefixes-grid", nargs="+", type=str, default=["true", "false"])
    args = p.parse_args()

    artifacts_root = Path(args.artifacts_root)
    snapshot_id = str(args.snapshot_id)

    corpus_path = artifacts_root / "data_snapshots" / snapshot_id / "retrieval_corpus.parquet"
    products_path = artifacts_root / "data_snapshots" / snapshot_id / "products.parquet"
    split_path = artifacts_root / "splits" / snapshot_id / "products_split.parquet"
    if not corpus_path.exists() or not products_path.exists() or not split_path.exists():
        raise FileNotFoundError("Missing retrieval corpus/products/splits under artifacts/data_snapshots and artifacts/splits.")

    corp = pd.read_parquet(corpus_path)
    prod = pd.read_parquet(products_path)[["product_id", "title"]]
    sp = pd.read_parquet(split_path)[["product_id", "split"]]
    corp = corp.merge(prod, on="product_id", how="left").merge(sp, on="product_id", how="left")
    corp = corp.dropna(subset=["product_document", "title", "split"]).reset_index(drop=True)

    docs = corp["product_document"].astype(str).tolist()
    titles = corp["title"].astype(str).tolist()
    val_indices = np.where(corp["split"].astype(str).to_numpy() == "val")[0]
    if len(val_indices) == 0:
        raise RuntimeError("No val rows found in products_split.parquet merge.")

    rng = np.random.default_rng(int(args.seed))
    if args.screening_n > 0 and len(val_indices) > args.screening_n:
        screen_indices = rng.choice(val_indices, size=int(args.screening_n), replace=False)
    else:
        screen_indices = val_indices
    if args.full_n > 0 and len(val_indices) > args.full_n:
        full_indices = rng.choice(val_indices, size=int(args.full_n), replace=False)
    else:
        full_indices = val_indices

    use_prefixes_grid = [str(x).lower() in ("1", "true", "yes", "y") for x in args.use_prefixes_grid]

    out_dir = artifacts_root / "evals" / snapshot_id / "retrieval" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = _utc_now_iso()
    all_dense_trials: list[DenseTrial] = []
    per_model_reports: dict[str, dict[str, Any]] = {}

    for model_name in args.models:
        model_trials: list[DenseTrial] = []
        t0 = time.time()
        for max_len in args.max_seq_lengths:
            for use_pref in use_prefixes_grid:
                # Encode docs + queries
                doc_emb, mdl, q_prefix = _encode_docs(
                    model_name=model_name,
                    docs=docs,
                    max_seq_length=int(max_len),
                    use_prefixes=bool(use_pref),
                    batch_size=int(args.batch_size),
                    device=str(args.device),
                )
                q_texts = [q_prefix + t for t in titles]
                q_emb = mdl.encode(
                    q_texts,
                    batch_size=int(args.batch_size),
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)

                ranks = _eval_self_match_ranks(doc_emb=doc_emb, q_emb=q_emb, indices=np.asarray(screen_indices, dtype=np.int64))
                trial = DenseTrial(
                    model=str(model_name),
                    max_seq_length=int(max_len),
                    use_prefixes=bool(use_pref),
                    recall_at_k=recall_at_k(ranks, int(args.k)),
                    ndcg_at_k=ndcg_at_k(ranks, int(args.k)),
                    mrr=mrr(ranks),
                    n_evaluated=int(len(ranks)),
                    phase="screening",
                )
                model_trials.append(trial)
                all_dense_trials.append(trial)

        # Refine: top-2 screening configs on full val (matches existing two-stage pattern).
        best_screen = _pick_best([t for t in model_trials if t.phase == "screening"])
        # Also pick the best alternative where use_prefixes differs (if present), to mirror reports.
        alt = None
        for t in model_trials:
            if t.phase != "screening":
                continue
            if t.max_seq_length == best_screen.max_seq_length and t.use_prefixes != best_screen.use_prefixes:
                alt = t
                break
        refine_candidates = [best_screen] + ([alt] if alt else [])

        refine_trials: list[DenseTrial] = []
        for cand in refine_candidates:
            doc_emb, mdl, q_prefix = _encode_docs(
                model_name=model_name,
                docs=docs,
                max_seq_length=int(cand.max_seq_length),
                use_prefixes=bool(cand.use_prefixes),
                batch_size=int(args.batch_size),
                device=str(args.device),
            )
            q_texts = [q_prefix + t for t in titles]
            q_emb = mdl.encode(
                q_texts,
                batch_size=int(args.batch_size),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            ranks = _eval_self_match_ranks(doc_emb=doc_emb, q_emb=q_emb, indices=np.asarray(full_indices, dtype=np.int64))
            refine_trials.append(
                DenseTrial(
                    model=str(model_name),
                    max_seq_length=int(cand.max_seq_length),
                    use_prefixes=bool(cand.use_prefixes),
                    recall_at_k=recall_at_k(ranks, int(args.k)),
                    ndcg_at_k=ndcg_at_k(ranks, int(args.k)),
                    mrr=mrr(ranks),
                    n_evaluated=int(len(ranks)),
                    phase="refine",
                )
            )
        elapsed = time.time() - t0

        # Write per-model report (schema_version=2 like existing)
        report = {
            "schema_version": 2,
            "snapshot_id": snapshot_id,
            "tuning_split": "val",
            "metrics_k": int(args.k),
            "two_stage": {
                "enabled": True,
                "screening_n_val_queries": int(len(screen_indices)),
                "full_n_val_queries": int(len(full_indices)),
                "refine_bm25_ran": False,
                "refine_dense_ran": True,
                "random_state": int(args.seed),
            },
            "started_at_utc": started,
            "bm25_trials": [],
            "dense_trials": [t.__dict__ for t in model_trials],
            "bm25_refine_trials": [],
            "dense_refine_trials": [t.__dict__ for t in refine_trials],
            "dense_model_filter": str(model_name),
            "ended_at_utc": _utc_now_iso(),
            "elapsed_s": float(elapsed),
            "selection_rule": "maximize (recall@k, nDCG@k, MRR) lexicographic; final pick from refine phase when present else screening",
        }
        safe_name = str(model_name).replace("/", "_")
        per_path = out_dir / f"tune_report_dense_{safe_name}.json"
        _write_json(per_path, report)
        per_model_reports[model_name] = report

    # Merge report (tune_report.json) and best params file (tuning_best_params.json)
    merged_trials = []
    merged_refine = []
    for rep in per_model_reports.values():
        merged_trials.extend(rep.get("dense_trials", []))
        merged_refine.extend(rep.get("dense_refine_trials", []))

    merged = dict(per_model_reports[next(iter(per_model_reports))]) if per_model_reports else {}
    merged["dense_trials"] = merged_trials
    merged["dense_refine_trials"] = merged_refine
    merged["dense_model_filter"] = None
    merged["ended_at_utc"] = _utc_now_iso()
    merged["merged_from"] = str(out_dir / "tuning_best_params.json")
    _write_json(out_dir / "tune_report.json", merged)

    # Select best per model from refine if present else screening.
    best_params = {}
    for model_name, rep in per_model_reports.items():
        refine = [DenseTrial(**t) for t in rep.get("dense_refine_trials", [])]
        screening = [DenseTrial(**t) for t in rep.get("dense_trials", [])]
        pick = _pick_best(refine) if refine else _pick_best(screening)
        best_params[str(model_name)] = {
            "max_seq_length": int(pick.max_seq_length),
            "use_prefixes": bool(pick.use_prefixes),
            "picked_phase": str(pick.phase),
            "val_recall_at_k": float(pick.recall_at_k),
            "val_ndcg_at_k": float(pick.ndcg_at_k),
            "val_mrr": float(pick.mrr),
        }
    _write_json(out_dir / "tuning_best_params.json", {"schema_version": 1, "snapshot_id": snapshot_id, "best_params": best_params, "written_at_utc": _utc_now_iso()})

    print(json.dumps({"ok": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

