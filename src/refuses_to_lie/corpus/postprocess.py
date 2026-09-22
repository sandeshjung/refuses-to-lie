"""Optional post-processing and inspection helpers."""

from __future__ import annotations

from refuses_to_lie.corpus.chunk import Chunk

_HISTOGRAM_BUCKETS: list[tuple[str, float]] = [
    ("0", 0),
    ("1-25", 25),
    ("26-75", 75),
    ("76-150", 150),
    ("151-300", 300),
    ("301-500", 500),
    ("500+", float("inf")),
]


def merge_short_chunks(chunks: list[Chunk], min_tokens: int, doc_id: str) -> list[Chunk]:
    """Merge any chunk below min_tokens into its next neighbor (or, for a
    trailing short chunk, into its previous neighbor). Off by default —
    see chunk_document's docstring for why."""
    merged: list[Chunk] = []
    for c in chunks:
        if merged and merged[-1].token_count < min_tokens:
            prev = merged.pop()
            c = _combine(prev, c, doc_id)
        merged.append(c)
    if len(merged) > 1 and merged[-1].token_count < min_tokens:
        last = merged.pop()
        prev = merged.pop()
        merged.append(_combine(prev, last, doc_id))
    for i, c in enumerate(merged, start=1):
        c.chunk_id = f"{doc_id}-{i:04d}"
    return merged


def _combine(first: Chunk, second: Chunk, doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=first.chunk_id,
        doc_id=doc_id,
        text=f"{first.text}\n{second.text}",
        heading_trail=first.heading_trail,
        section_no=first.section_no,
        page_start=first.page_start,
        page_end=second.page_end,
        token_count=first.token_count + second.token_count,
    )


def token_histogram(chunks: list[Chunk]) -> dict[str, int]:
    """Bucket chunk token counts so the real distribution can be inspected
    before ever turning merging on."""
    buckets = {label: 0 for label, _ in _HISTOGRAM_BUCKETS}
    for c in chunks:
        for label, upper_bound in _HISTOGRAM_BUCKETS:
            if c.token_count <= upper_bound:
                buckets[label] += 1
                break
    return buckets
