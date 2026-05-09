"""Dense retrieval over retrieval_corpus using intfloat/e5-large-v2.

On startup, loads the pre-built FAISS index from:
  artifacts/indexes/{snapshot_id}/dense/intfloat_e5-large-v2/

If no saved index is found, call build_index() to encode the corpus and
save the index to disk for future startups.

At query time, only the user query is encoded (fast, single text).
The product corpus is never re-encoded at runtime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_ID = "38710839ca6e1009"
MODEL_NAME = "intfloat/e5-large-v2"
MAX_SEQ_LENGTH = 384
USE_PREFIXES = True
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass
class RetrievalConfig:
    model_name: str = MODEL_NAME
    max_seq_length: int = MAX_SEQ_LENGTH
    use_prefixes: bool = USE_PREFIXES
    top_k: int = 10
    min_score: float = 0.0
    device: str = "cpu"


@dataclass
class RetrievalAgent:
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    snapshot_id: str = SNAPSHOT_ID
    artifacts_root: Path | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)
    _corpus: list[dict[str, Any]] = field(default_factory=list, init=False)

    def _index_dir(self) -> Path:
        root = self.artifacts_root or Path(__file__).resolve().parents[2] / "artifacts"
        return root / "indexes" / self.snapshot_id / "dense" / "intfloat_e5-large-v2"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer, models

        device = (
            os.getenv("COPILOT_RETRIEVAL_DEVICE")
            or os.getenv("COPILOT_V2_RETRIEVER_DEVICE")
            or self.config.device
            or "cpu"
        )
        try:
            self._model = SentenceTransformer(self.config.model_name, device=device)
        except TypeError as e:
            # Some local HF cache snapshots can have the Transformer weights but miss
            # the SentenceTransformers pooling config. E5 uses mean pooling, so build
            # the equivalent runtime model without forcing a metadata download.
            if "Pooling.__init__" not in str(e) or "e5" not in self.config.model_name.lower():
                raise
            word_embedding_model = models.Transformer(
                self.config.model_name,
                max_seq_length=self.config.max_seq_length,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            self._model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        self._model.max_seq_length = self.config.max_seq_length

    def load_index(self) -> bool:
        """Load pre-built FAISS index from disk. Returns True if successful."""
        import faiss
        import pandas as pd

        idx_dir = self._index_dir()
        faiss_path = idx_dir / "index_flatip.faiss"
        corpus_path = idx_dir / "corpus.parquet"

        if not faiss_path.exists() or not corpus_path.exists():
            return False

        print(f"Loading saved index from {idx_dir}")
        self._index = faiss.read_index(str(faiss_path))
        corpus_df = pd.read_parquet(corpus_path)
        self._corpus = corpus_df.to_dict(orient="records")
        self._load_model()
        print(f"  Loaded {len(self._corpus)} products, dim={self._index.d}")
        return True

    def build_index(self, corpus: list[dict[str, Any]], text_field: str = "product_document", save: bool = True) -> None:
        """Encode corpus and build FAISS index. Saves to disk by default."""
        import faiss
        import pandas as pd

        self._load_model()
        self._corpus = corpus
        texts = [doc.get(text_field, "") for doc in corpus]
        if self.config.use_prefixes:
            texts = [PASSAGE_PREFIX + t for t in texts]

        print(f"Encoding {len(texts)} documents...")
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        if save:
            self._save_index(embeddings, pd.DataFrame(corpus))

    def _save_index(self, embeddings: np.ndarray, corpus_df: Any) -> None:
        import faiss

        idx_dir = self._index_dir()
        idx_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(idx_dir / "index_flatip.faiss"))
        corpus_df.to_parquet(idx_dir / "corpus.parquet", index=False)
        np.save(str(idx_dir / "doc_emb_norm_f32.npy"), embeddings)

        meta = {
            "schema_version": 1,
            "index_kind": "dense_faiss_flatip",
            "snapshot_id": self.snapshot_id,
            "model_name": self.config.model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "n_docs": len(self._corpus),
            "use_prefixes": self.config.use_prefixes,
            "document_prefix": PASSAGE_PREFIX,
            "query_prefix": QUERY_PREFIX,
            "max_seq_length": self.config.max_seq_length,
        }
        (idx_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved index to {idx_dir}")

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Encode query and return top-k results from the index."""
        if self._index is None or self._model is None:
            raise RuntimeError("No index loaded. Call load_index() or build_index() first.")
        q = (QUERY_PREFIX + query) if self.config.use_prefixes else query
        q_emb = self._model.encode([q], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        k = int(top_k) if top_k is not None else int(self.config.top_k)
        scores, indices = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < float(self.config.min_score or 0.0):
                continue
            entry = dict(self._corpus[idx])
            entry["retrieval_score"] = float(score)
            results.append(entry)
        return results
