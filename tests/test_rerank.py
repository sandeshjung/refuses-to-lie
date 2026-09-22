import pytest

from refuses_to_lie.corpus import Chunk
from refuses_to_lie.rerank import Reranker, get_reranker
from refuses_to_lie.retrieval import Hit


def _hit(chunk_id: str, text: str, score: float) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            text=text,
            heading_trail=("Section",),
            section_no="1",
            page_start=1,
            page_end=1,
            token_count=len(text.split()),
        ),
        score=score,
    )


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return get_reranker()


def test_reranker_promotes_the_relevant_chunk_over_a_higher_scored_distractor(
    reranker: Reranker,
):
    # The distractor arrives ranked first with a better first-stage score.
    # A cross-encoder reads query and passage together, so it should catch
    # what a bi-encoder missed and push the answer-bearing chunk to the top.
    hits = [
        _hit("distractor", "The annual leave year runs from 1 April to 31 March.", 0.9),
        _hit(
            "relevant",
            "Employees must have a minimum of 12 months continuous service with the "
            "Trust before they may apply for a career break.",
            0.4,
        ),
    ]
    reranked = reranker.rerank("how much service is needed to apply for a career break", hits)

    assert [h.chunk.chunk_id for h in reranked] == ["relevant", "distractor"]


def test_reranking_preserves_the_candidate_set(reranker: Reranker):
    hits = [
        _hit(f"c{i}", f"Policy clause number {i} about leave.", 1.0 - i / 10) for i in range(5)
    ]
    reranked = reranker.rerank("leave policy", hits)

    assert {h.chunk.chunk_id for h in reranked} == {h.chunk.chunk_id for h in hits}
    assert len(reranked) == len(hits)


def test_reranking_returns_scores_in_descending_order(reranker: Reranker):
    hits = [
        _hit(f"c{i}", f"Clause {i} concerning sick pay entitlement.", 0.5) for i in range(4)
    ]
    scores = [h.score for h in reranker.rerank("sick pay", hits)]

    assert scores == sorted(scores, reverse=True)


def test_empty_shortlist_is_handled_without_calling_the_model(reranker: Reranker):
    assert reranker.rerank("anything", []) == []


def test_get_reranker_caches_the_loaded_model():
    # Loading costs seconds; a grid run over hundreds of questions must pay
    # it once, not once per question.
    assert get_reranker() is get_reranker()
