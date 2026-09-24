from refuses_to_lie.grading import (
    UNGRADED,
    Grade,
    _parse_grade,
    accuracy,
    grade_cache_key,
    gradeable,
)


def _row(**extra) -> dict:
    row = {
        "question_id": "AS-001",
        "config_id": "A",
        "question": "How much annual leave?",
        "expected_answer_summary": "27 days plus 8 public holidays.",
        "answer": "27 days plus 8 public holidays [DOC-1-0001].",
        "expected_answerable": True,
        "abstained": False,
    }
    row.update(extra)
    return row


def test_incorrect_is_parsed_before_correct():
    # "INCORRECT" contains "CORRECT", so checking in the wrong order reads
    # every wrong answer as right -- silently halving the reported error.
    assert _parse_grade("INCORRECT") == "INCORRECT"
    assert _parse_grade("CORRECT") == "CORRECT"
    assert _parse_grade("The answer is INCORRECT.") == "INCORRECT"


def test_partial_is_parsed():
    assert _parse_grade("PARTIAL") == "PARTIAL"


def test_an_unparseable_reply_is_not_silently_treated_as_correct():
    assert _parse_grade("I'm not sure how to grade this") == UNGRADED


def test_partial_counts_as_wrong():
    # A half-right entitlement answer is still actionable and still wrong.
    assert Grade("Q", "A", "PARTIAL", "text").is_wrong
    assert Grade("Q", "A", "INCORRECT", "text").is_wrong
    assert not Grade("Q", "A", "CORRECT", "text").is_wrong


def test_abstentions_are_never_graded():
    # Refusing is the behaviour the project rewards; grading it for
    # correctness would score the system's best moments as failures.
    assert not gradeable(_row(abstained=True))


def test_unanswerable_questions_are_not_graded():
    # There is no reference answer to compare against; error_floor covers it.
    assert not gradeable(_row(expected_answerable=False))


def test_failed_rows_are_not_graded():
    assert not gradeable(_row(error="APIError: 429"))


def test_rows_without_a_reference_answer_are_not_graded():
    assert not gradeable(_row(expected_answer_summary=""))


def test_a_normal_answered_row_is_gradeable():
    assert gradeable(_row())


def test_identical_answers_share_a_cache_key():
    # Later rungs often leave the text untouched; grading it twice would
    # pay for the same judgement on every rung of the ladder.
    same = grade_cache_key(_row(config_id="A"))
    assert same == grade_cache_key(_row(config_id="F"))


def test_a_changed_answer_gets_a_new_cache_key():
    assert grade_cache_key(_row()) != grade_cache_key(_row(answer="25 days."))


def test_accuracy_excludes_ungraded_from_both_sides():
    # A parsing failure must not inflate either correctness or error.
    grades = [
        Grade("Q1", "A", "CORRECT", ""),
        Grade("Q2", "A", "INCORRECT", ""),
        Grade("Q3", "A", UNGRADED, ""),
    ]
    correct, wrong = accuracy(grades)
    assert correct == 0.5
    assert wrong == 0.5


def test_accuracy_of_nothing_is_zero_not_a_crash():
    assert accuracy([]) == (0.0, 0.0)
