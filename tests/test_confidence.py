from pathlib import Path

import pytest

from refuses_to_lie.confidence import (
    AgreementResult,
    ConfidenceSignal,
    compute_confidence,
    retrieval_confidence,
    sample_agreement,
    should_abstain,
)
from refuses_to_lie.config import A, D
from refuses_to_lie.corpus import Chunk, chunk_document, load_body_pages
from refuses_to_lie.generation import GeneratedAnswer
from refuses_to_lie.retrieval import Hit, Index
from refuses_to_lie.verifier import ClaimVerdict, VerifiedAnswer

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


def _fake_hit(chunk_id: str, score: float) -> Hit:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text="irrelevant for this test",
        heading_trail=("Section",),
        section_no="1",
        page_start=1,
        page_end=1,
        token_count=5,
    )
    return Hit(chunk=chunk, score=score)


def test_retrieval_confidence_high_margin_is_near_1():
    hits = [_fake_hit("a", 0.95), _fake_hit("b", 0.10), _fake_hit("c", 0.05)]
    assert retrieval_confidence(hits) == pytest.approx((0.95 - 0.10) / (0.95 - 0.05))


def test_retrieval_confidence_identical_scores_is_neutral():
    hits = [_fake_hit("a", 0.5), _fake_hit("b", 0.5), _fake_hit("c", 0.5)]
    assert retrieval_confidence(hits) == 0.5


def test_retrieval_confidence_single_hit_is_1():
    assert retrieval_confidence([_fake_hit("a", 0.1)]) == 1.0


def test_retrieval_confidence_no_hits_is_0():
    assert retrieval_confidence([]) == 0.0


def test_agreement_result_score_all_abstained_is_agreement():
    result = AgreementResult(
        samples=[], citation_overlap=0.0, any_abstained=True, all_abstained=True
    )
    assert result.score == 1.0


def test_agreement_result_score_partial_abstention_is_disagreement():
    result = AgreementResult(
        samples=[], citation_overlap=0.9, any_abstained=True, all_abstained=False
    )
    assert result.score == 0.0


def test_agreement_result_score_falls_back_to_citation_overlap():
    result = AgreementResult(
        samples=[], citation_overlap=0.6, any_abstained=False, all_abstained=False
    )
    assert result.score == 0.6


def test_sample_agreement_requires_citations():
    with pytest.raises(ValueError):
        sample_agreement("question", [_fake_hit("a", 1.0)], A)


def test_compute_confidence_weighted_composite():
    verified = VerifiedAnswer(
        answer=GeneratedAnswer(question="q", text="a", citations=[], context_chunk_ids=[]),
        claim_verdicts=[ClaimVerdict("claim", "doc-0001", "SUPPORTED")],
    )
    agreement = AgreementResult(
        samples=[], citation_overlap=1.0, any_abstained=False, all_abstained=False
    )
    hits = [_fake_hit("a", 1.0), _fake_hit("b", 0.0)]

    signal = compute_confidence(hits, verified, agreement)

    assert signal.retrieval == 1.0
    assert signal.groundedness == 1.0
    assert signal.agreement == 1.0
    assert signal.composite == pytest.approx(1.0)


def test_compute_confidence_custom_weights():
    verified = VerifiedAnswer(
        answer=GeneratedAnswer(question="q", text="a", citations=[], context_chunk_ids=[]),
        claim_verdicts=[],
    )
    agreement = AgreementResult(
        samples=[], citation_overlap=0.0, any_abstained=False, all_abstained=False
    )
    hits = [_fake_hit("a", 1.0)]

    weights = {"retrieval": 1.0, "groundedness": 0.0, "agreement": 0.0}
    signal = compute_confidence(hits, verified, agreement, weights=weights)

    assert signal.composite == pytest.approx(1.0)


def test_should_abstain_below_threshold():
    signal = ConfidenceSignal(retrieval=0.1, groundedness=0.1, agreement=0.1)
    assert should_abstain(signal, threshold=0.5) is True


def test_should_abstain_at_or_above_threshold():
    signal = ConfidenceSignal(retrieval=0.9, groundedness=0.9, agreement=0.9)
    assert should_abstain(signal, threshold=0.5) is False


@pytest.mark.live
def test_sample_agreement_real_question_gets_consistent_citations():
    chunks = chunk_document(
        "UHN-PO-HR24",
        load_body_pages(CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"),
    )
    index = Index(chunks)
    hits = index.search_hybrid("Can unused annual leave be carried into the next year?", k=4)

    agreement = sample_agreement(
        "Can unused annual leave be carried into the next year?",
        hits,
        D,
        cache_key_prefix="test-confidence-agreement",
    )

    assert len(agreement.samples) == D.agreement_samples
    assert 0.0 <= agreement.citation_overlap <= 1.0
    # A stable, clearly-answerable question at config D's low temperature
    # (0.2) should not produce split abstention across samples.
    assert agreement.any_abstained == agreement.all_abstained
