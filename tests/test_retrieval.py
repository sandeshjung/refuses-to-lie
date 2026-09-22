from pathlib import Path

import pytest

from refuses_to_lie.corpus import chunk_document, load_body_pages
from refuses_to_lie.retrieval import Index

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


@pytest.fixture(scope="module")
def index() -> Index:
    chunks = []
    chunks += chunk_document(
        "a36", load_body_pages(CORPUS / "a36-employment-break-policy-april-26.pdf")
    )
    chunks += chunk_document(
        "UHN-PO-HR24",
        load_body_pages(CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"),
    )
    return Index(chunks)


def test_bm25_finds_exact_phrase(index: Index):
    hits = index.search_bm25("months continuous service with the trust", k=3)

    assert hits
    assert hits[0].chunk.doc_id == "a36"
    assert hits[0].chunk.section_no == "1"
    assert "continuous service" in hits[0].chunk.text


def test_dense_finds_semantic_paraphrase_with_no_keyword_overlap(index: Index):
    # No shared vocabulary with the source text ("annual leave year", "1st
    # April to 31st March") beyond the single word "year" — this has to be
    # matched on meaning, not term overlap, which is exactly what BM25 can't
    # do and dense retrieval is for.
    hits = index.search_dense(
        "when does the yearly holiday allowance period start and end", k=5
    )

    assert hits
    doc_ids = [h.chunk.doc_id for h in hits]
    assert doc_ids.count("UHN-PO-HR24") >= 3

    section_nos = {h.chunk.section_no for h in hits}
    assert "4.1" in section_nos or "4.2" in section_nos


def test_hybrid_still_surfaces_the_bm25_match(index: Index):
    hits = index.search_hybrid("months continuous service with the trust", k=5)

    chunk_ids = [h.chunk.chunk_id for h in hits]
    assert any(h.chunk.doc_id == "a36" and h.chunk.section_no == "1" for h in hits)
    assert len(chunk_ids) == len(set(chunk_ids))  # no duplicate chunks


def test_search_respects_k(index: Index):
    assert len(index.search_bm25("annual leave", k=2)) == 2
    assert len(index.search_dense("annual leave", k=2)) == 2
    assert len(index.search_hybrid("annual leave", k=2)) == 2
