"""Dense + lexical retrieval over retrieval_corpus using intfloat/e5-large-v2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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


@dataclass
class RetrievalAgent:
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    _model: Any = field(default=None, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)
    _corpus: list[dict[str, Any]] = field(default_factory=list, init=False)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.config.model_name)
        self._model.max_seq_length = self.config.max_seq_length

    def build_index(self, corpus: list[dict[str, Any]], text_field: str = "product_document") -> None:
        """Encode corpus documents and build an in-memory FAISS index."""
        import faiss

        self._load_model()
        self._corpus = corpus
        texts = [doc.get(text_field, "") for doc in corpus]
        if self.config.use_prefixes:
            texts = [PASSAGE_PREFIX + t for t in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return top-k corpus entries for a query."""
        if self._index is None or self._model is None:
            raise RuntimeError("Call build_index() before retrieve().")
        q = (QUERY_PREFIX + query) if self.config.use_prefixes else query
        q_emb = self._model.encode([q], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self._index.search(q_emb, self.config.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self._corpus[idx])
            entry["retrieval_score"] = float(score)
            results.append(entry)
        return results
