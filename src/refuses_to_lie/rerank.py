"""Cross-encoder reranking — the ladder's rung C.

Bi-encoder retrieval (retrieval.py) embeds the query and every chunk
separately, so it never actually compares them; it compares two summaries
of them. A cross-encoder reads the query and one chunk *together* and
scores that pair directly, which is far more accurate and far too slow to
run over a whole corpus. So it runs second, over the shortlist the fast
rankers produced.

A note on the scores, because it affects how they can be used: this model
emits unbounded logits, and on this corpus they are mostly negative even
for the correct chunk — policy prose scores much lower than the web text
the model was trained on. They are meaningful as an ORDERING and not as a
magnitude, so nothing here should compare a reranked score against a fixed
number. retrieval_confidence() is safe because it only ever looks at
differences between scores within one query's results.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from refuses_to_lie.retrieval import Hit

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Wraps one loaded cross-encoder. Loading costs seconds, scoring a
    shortlist costs milliseconds, so instances are meant to be reused —
    see get_reranker()."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Rescore and reorder `hits` against `query`.

        Returns the same chunks, never a different set — reranking reorders
        a shortlist, it cannot recover a chunk the first-stage rankers
        missed. That ceiling is the point of measuring rung C separately
        from rung B.
        """
        if not hits:
            return []
        scores = self._model.predict([(query, hit.chunk.text) for hit in hits])
        rescored = [
            Hit(chunk=hit.chunk, score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> Reranker:
    """Process-wide cached loader, so a grid run over hundreds of questions
    pays the model load once rather than once per question."""
    return Reranker(model_name)
