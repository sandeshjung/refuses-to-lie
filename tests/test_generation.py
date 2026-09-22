from pathlib import Path

import pytest

from refuses_to_lie.config import A, D
from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.generation import GeneratedAnswer, generate_answer
from refuses_to_lie.retrieval import Index

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = chunk_document(
        "UHN-PO-HR24",
        load_body_pages(CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"),
    )
    return Index(chunks)


def test_generate_answer_with_citations(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)

    answer = generate_answer(
        "Can unused annual leave be carried into the next year?",
        hits,
        D,
        cache_key="test-generation-with-citations",
    )

    assert answer.text.strip()
    assert answer.citations
    context_ids = set(answer.context_chunk_ids)
    for citation in answer.citations:
        assert citation.chunk_id in context_ids
        assert citation.cite_label.startswith("UHN-PO-HR24")


def test_generate_answer_without_citations_extracts_none(index: Index):
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)

    answer = generate_answer(
        "Can unused annual leave be carried into the next year?",
        hits,
        A,
        cache_key="test-generation-without-citations",
    )

    assert answer.text.strip()
    assert answer.citations == []


def test_generate_answer_requires_at_least_one_hit(index: Index):
    with pytest.raises(ValueError):
        generate_answer("anything", [], D)


def test_abstained_property():
    abstained = GeneratedAnswer(
        question="q",
        text="I don't have enough information in the provided documents to answer this.",
        citations=[],
        context_chunk_ids=[],
    )
    answered = GeneratedAnswer(
        question="q", text="Yes, per section 4.8.", citations=[], context_chunk_ids=[]
    )

    assert abstained.abstained is True
    assert answered.abstained is False
