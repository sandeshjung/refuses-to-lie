from pathlib import Path

import pytest

from refuses_to_lie.answering import answer_question, retrieve_context
from refuses_to_lie.config import LADDER, A, B, C, E, F
from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.retrieval import Index

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = chunk_document(
        "a36", load_body_pages(CORPUS / "a36-employment-break-policy-april-26.pdf")
    )
    return Index(chunks)


def test_retrieval_axis_selects_a_different_ranker(index: Index):
    # A and B differ only on the retrieval axis, so if the orchestrator were
    # ignoring config.retrieval this would silently return identical hits and
    # the A->B rung of the ablation would measure nothing.
    question = "how long must someone work here before taking a career break"
    dense = retrieve_context(index, question, A)
    hybrid = retrieve_context(index, question, B)

    assert [h.chunk.chunk_id for h in dense] != [h.chunk.chunk_id for h in hybrid] or [
        h.score for h in dense
    ] != [h.score for h in hybrid]


def test_context_is_narrowed_to_top_k_context(index: Index):
    hits = retrieve_context(index, "career break eligibility", A)
    assert len(hits) <= A.top_k_context
    assert A.top_k_context < A.top_k_retrieve


def test_rerank_rung_reorders_without_changing_the_candidate_set(index: Index):
    # Reranking must reorder the shortlist, not fetch from the corpus again:
    # rung C can only improve on what rung B already surfaced, and that
    # ceiling is what makes B->C a meaningful thing to measure separately.
    question = "how long must someone work here before taking a career break"
    before = retrieve_context(index, question, B)
    after = retrieve_context(index, question, C)

    shortlist = {h.chunk.chunk_id for h in index.search_hybrid(question, k=B.top_k_retrieve)}
    assert {h.chunk.chunk_id for h in after} <= shortlist
    assert len(after) == len(before)
    # Scores come from the cross-encoder now, so they are on a different
    # scale entirely (unbounded logits, often negative) rather than RRF sums.
    assert [h.score for h in after] != [h.score for h in before]
    assert after == sorted(after, key=lambda h: h.score, reverse=True)


def test_abstention_without_a_verifier_is_rejected(index: Index):
    from dataclasses import replace

    broken = replace(F, verifier="off")
    with pytest.raises(ValueError, match="verifier"):
        answer_question("anything", index, broken)


def test_ladder_rungs_turn_stages_on_cumulatively():
    # The orchestrator reads these flags directly, so this pins the shape it
    # depends on: once a stage is on it stays on for every later rung.
    assert A.retrieval == "dense" and B.retrieval == "hybrid"
    assert not A.require_citations and LADDER[3].require_citations
    assert E.verifier == "drop_unsupported" and F.verifier == "drop_unsupported"
    assert not E.abstain and F.abstain
    assert F.require_citations, "abstention measures citation agreement, so D must stay on"
