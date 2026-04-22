from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from copilot_v2.runtime.orchestrator import _default_owner_for_product


def _write_index_meta(
    *,
    out_path: Path,
    snapshot_id: str,
    owner_id: str,
    model_name: str,
    embedding_dim: int,
    n_docs: int,
    use_prefixes: bool,
    query_prefix: str,
    document_prefix: str,
    max_seq_length: int,
    batch_size_encode: int,
    device: str,
    corpus_parquet: Path,
    faiss_index: Path,
    doc_embeddings_npy: Path,
) -> None:
    base_dir = out_path.parent
    payload: dict[str, Any] = {
        "schema_version": 1,
        "index_kind": "dense_faiss_flatip",
        "snapshot_id": snapshot_id,
        "owner_id": owner_id,
        "model_name": model_name,
        "embedding_dim": int(embedding_dim),
        "n_docs": int(n_docs),
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
        "use_prefixes": bool(use_prefixes),
        "max_seq_length": int(max_seq_length),
        "batch_size_encode": int(batch_size_encode),
        "device": str(device),
        "id_col": "product_id",
        "doc_col": "product_document",
        "paths": {
            # Store relative paths for portability (resolved relative to index_meta.json location).
            "corpus_parquet": str(Path(corpus_parquet).resolve().relative_to(base_dir.resolve())),
            "faiss_index": str(Path(faiss_index).resolve().relative_to(base_dir.resolve())),
            "doc_embeddings_npy": str(Path(doc_embeddings_npy).resolve().relative_to(base_dir.resolve())),
        },
        "build": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sentence_transformers_version": getattr(__import__("sentence_transformers"), "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "note": "Owner-scoped dense index. Embeddings are L2-normalized; inner product equals cosine similarity.",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_owner_indexes(
    *,
    snapshot_id: str,
    artifacts_root: Path,
    model_name: str,
    n_owners: int,
    max_seq_length: int,
    batch_size_encode: int,
    device: str,
) -> None:
    corpus_path = artifacts_root / "data_snapshots" / snapshot_id / "retrieval_corpus.parquet"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing retrieval corpus: {corpus_path}")

    corpus = pd.read_parquet(corpus_path)
    if "product_id" not in corpus.columns or "product_document" not in corpus.columns:
        raise ValueError("retrieval_corpus.parquet must contain product_id and product_document")

    corpus["owner_id"] = corpus["product_id"].astype(str).map(lambda pid: _default_owner_for_product(pid, n_owners=int(n_owners)))

    model = SentenceTransformer(str(model_name))
    try:
        model.max_seq_length = int(max_seq_length)
    except Exception:
        pass

    use_prefixes = True
    query_prefix = "query: "
    document_prefix = "passage: "

    root_out = artifacts_root / "indexes" / snapshot_id / "owners"
    root_out.mkdir(parents=True, exist_ok=True)

    for owner_id, grp in corpus.groupby("owner_id", sort=True):
        owner_id = str(owner_id)
        out_dir = root_out / owner_id / "dense" / "intfloat_e5-large-v2"
        out_dir.mkdir(parents=True, exist_ok=True)

        owner_corpus = grp[["product_id", "product_document"]].copy().reset_index(drop=True)
        owner_corpus_path = out_dir / "corpus.parquet"
        owner_corpus.to_parquet(owner_corpus_path, index=False)

        docs = owner_corpus["product_document"].astype(str).tolist()
        # Encode with normalization; we use IP index for cosine.
        emb = model.encode(
            [f"{document_prefix}{d}" if use_prefixes else d for d in docs],
            normalize_embeddings=True,
            batch_size=int(batch_size_encode),
            show_progress_bar=True,
            device=str(device),
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError("Expected 2D embeddings array")

        dim = int(emb.shape[1])
        idx = faiss.IndexFlatIP(dim)
        idx.add(emb)

        faiss_path = out_dir / "index_flatip.faiss"
        faiss.write_index(idx, str(faiss_path))

        emb_path = out_dir / "doc_emb_norm_f32.npy"
        np.save(str(emb_path), emb)

        meta_path = out_dir / "index_meta.json"
        _write_index_meta(
            out_path=meta_path,
            snapshot_id=snapshot_id,
            owner_id=owner_id,
            model_name=str(model_name),
            embedding_dim=dim,
            n_docs=len(owner_corpus),
            use_prefixes=use_prefixes,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            max_seq_length=int(max_seq_length),
            batch_size_encode=int(batch_size_encode),
            device=str(device),
            corpus_parquet=owner_corpus_path,
            faiss_index=faiss_path,
            doc_embeddings_npy=emb_path,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", type=str, required=True)
    ap.add_argument("--artifacts-root", type=Path, default=Path("copilot-v2/artifacts"))
    ap.add_argument("--model", type=str, default="intfloat/e5-large-v2")
    ap.add_argument("--n-owners", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=384)
    ap.add_argument("--batch-size-encode", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    build_owner_indexes(
        snapshot_id=str(args.snapshot_id),
        artifacts_root=Path(args.artifacts_root),
        model_name=str(args.model),
        n_owners=int(args.n_owners),
        max_seq_length=int(args.max_seq_length),
        batch_size_encode=int(args.batch_size_encode),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()

