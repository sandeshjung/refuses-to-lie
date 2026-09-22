"""Measure the retrieval rungs of the ladder (A, B, C) against the eval set.

Retrieval-only, so it makes no LLM calls and costs nothing to re-run: every
answerable eval question records which document SHOULD be cited, so we can
ask whether that document reached the context window the generator would
have seen.

This supersedes the hand-written query cases in eval_retrieval.py, which
predate the eval set and cover a handful of questions rather than 164.

Usage: uv run python scripts/compare_ladder_retrieval.py [sample_size]
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from refuses_to_lie.answering import retrieve_context
from refuses_to_lie.config import A, B, C, RunConfig
from refuses_to_lie.pipeline import load_corpus_chunks
from refuses_to_lie.rerank import Reranker, get_reranker
from refuses_to_lie.retrieval import Index

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Result:
    label: str
    top1: float
    hit_at_k: float
    mrr: float
    k: int


def score(
    config: RunConfig,
    label: str,
    questions: list[dict],
    index: Index,
    reranker: Reranker,
) -> Result:
    top1 = hits_at_k = 0
    mrr = 0.0
    for question in questions:
        wanted = set(question["expected_doc_ids"])
        retrieved = retrieve_context(index, question["question"], config, reranker=reranker)
        doc_ids = [hit.chunk.doc_id for hit in retrieved]

        if doc_ids and doc_ids[0] in wanted:
            top1 += 1
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id in wanted:
                hits_at_k += 1
                mrr += 1.0 / rank
                break

    n = len(questions)
    return Result(label, top1 / n, hits_at_k / n, mrr / n, config.top_k_context)


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    answerable = [q for q in questions if q["expected_answerable"] and q["expected_doc_ids"]]
    if sample_size:
        random.seed(0)
        answerable = random.sample(answerable, min(sample_size, len(answerable)))

    chunks = load_corpus_chunks(ROOT / "corpus" / "employer", ROOT / "corpus" / "statutory")
    index = Index(chunks)
    reranker = get_reranker()
    print(f"{len(chunks)} chunks | {len(answerable)} answerable questions\n")

    results = [
        score(A, "A  dense", answerable, index, reranker),
        score(B, "B  hybrid", answerable, index, reranker),
        score(C, "C  hybrid + rerank", answerable, index, reranker),
    ]
    for r in results:
        print(f"{r.label:22} top1={r.top1:6.1%}  hit@{r.k}={r.hit_at_k:6.1%}  mrr={r.mrr:.3f}")

    print(
        "\nNote: hit@k measures DOCUMENT-level recall. The eval set records no "
        "chunk-level ground truth, so this cannot show whether reranking surfaces a "
        "better passage within an already-correct document."
    )


if __name__ == "__main__":
    main()
