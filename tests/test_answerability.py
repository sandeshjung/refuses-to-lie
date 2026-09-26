from refuses_to_lie.answerability import (
    SCORES,
    UNKNOWN,
    AnswerabilitySignal,
    parse_verdict,
)


def test_unanswerable_is_not_misread_as_answerable():
    # "UNANSWERABLE" contains "ANSWERABLE". Checking the short label first
    # would read every refusal from the gate as a green light.
    assert parse_verdict("UNANSWERABLE") == "UNANSWERABLE"
    assert parse_verdict("The context is unanswerable.") == "UNANSWERABLE"
    assert parse_verdict("ANSWERABLE") == "ANSWERABLE"


def test_partial_is_parsed():
    assert parse_verdict("PARTIAL") == "PARTIAL"


def test_an_unparseable_reply_fails_closed():
    # When the gate cannot tell, the system must refuse, not answer.
    verdict = parse_verdict("I think it probably covers it?")
    assert verdict == UNKNOWN
    assert SCORES.get(verdict, 0.0) == 0.0


def test_only_a_full_answerable_clears_rung_gs_threshold():
    from refuses_to_lie.config import G

    assert SCORES["ANSWERABLE"] >= G.abstain_threshold
    assert SCORES["PARTIAL"] < G.abstain_threshold
    assert SCORES["UNANSWERABLE"] < G.abstain_threshold


def test_signal_exposes_composite_so_thresholding_is_shared():
    assert AnswerabilitySignal("PARTIAL", 0.5).composite == 0.5
