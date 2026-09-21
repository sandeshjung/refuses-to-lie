"""BM25 + dense retrieval over a fixed list of Chunks, combined via
reciprocal rank fusion (RRF) for hybrid search.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from refuses_to_lie.corpus import Chunk

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Hit:
    __slots__ = ("chunk", "score")

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score

    def __repr__(self) -> str:
        return f"Hit(chunk_id={self.chunk.chunk_id!r}, score={self.score:.4f})"


class Index:
    """Builds both a BM25 index and dense embeddings over `chunks` once, so
    search_bm25 / search_dense / search_hybrid are all cheap per query."""

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_EMBEDDING_MODEL):
        if not chunks:
            raise ValueError("Index requires at least one chunk")
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._model = SentenceTransformer(model_name)
        self._embeddings = self._model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def search_bm25(self, query: str, k: int = 20) -> list[Hit]:
        scores = self._bm25.get_scores(_tokenize(query))
        return self._top_k(scores, k)

    def search_dense(self, query: str, k: int = 20) -> list[Hit]:
        query_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self._embeddings @ query_vec
        return self._top_k(scores, k)

    def search_hybrid(self, query: str, k: int = 20, rrf_k: int = 60) -> list[Hit]:
        # Retrieve deeper than k from each ranker before fusing, so a chunk
        # ranked highly by only one of the two still has a chance to surface.
        pool = max(k * 2, 50)
        bm25_hits = self.search_bm25(query, k=pool)
        dense_hits = self.search_dense(query, k=pool)

        rrf_scores: dict[str, float] = {}
        for hits in (bm25_hits, dense_hits):
            for rank, hit in enumerate(hits):
                rrf_scores[hit.chunk.chunk_id] = rrf_scores.get(
                    hit.chunk.chunk_id, 0.0
                ) + 1.0 / (rrf_k + rank + 1)

        by_id = {c.chunk_id: c for c in self.chunks}
        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(chunk=by_id[chunk_id], score=score) for chunk_id, score in ranked]

    def _top_k(self, scores: np.ndarray, k: int) -> list[Hit]:
        k = min(k, len(self.chunks))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top_idx]
