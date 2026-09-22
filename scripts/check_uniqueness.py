"""Uniqueness-checking workflow for near-miss and out-of-scope eval
questions: a near-miss question must not be answerable by ANY chunk in the
corpus other than the one it's deliberately missing the fact from, and an
out-of-scope question must not be answerable by any chunk at all.

Builds the full-corpus index once, then runs each candidate question
through hybrid search and prints the top hits for manual review — this is
a review aid, not an automatic pass/fail, since "does this chunk actually
answer the question" needs a human to read the excerpt.

Usage: uv run python scripts/check_uniqueness.py "question one" "question two" ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from refuses_to_lie.pipeline import load_corpus_chunks
from refuses_to_lie.retrieval import Index

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    questions = sys.argv[1:]
    if not questions:
        print('Usage: uv run python scripts/check_uniqueness.py "question" [...]')
        return

    print("Loading corpus and building index...")
    chunks = load_corpus_chunks(ROOT / "corpus" / "employer", ROOT / "corpus" / "statutory")
    index = Index(chunks)
    print(f"{len(chunks)} chunks indexed\n")

    for q in questions:
        print("=" * 100)
        print(q)
        print("=" * 100)
        for h in index.search_hybrid(q, k=8):
            print(f"  {h.score:.4f}  {h.chunk.cite_label}")
            print(f"    {h.chunk.text[:200].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
