"""Judged correctness: was the answer actually right?

The grid's own metrics can only prove an answer wrong when the question
had no answer to begin with, or when an adversarial document was cited.
Neither catches the failure the project is really about -- a fluent,
correctly-cited answer carrying the wrong number. Deciding that needs a
reader, so this module asks one.

Three deliberate choices:

  - The judge is the VERIFIER's model, not the generator's. A model
    grading its own output agrees with itself, and the resulting number
    would measure self-consistency rather than correctness.
  - It grades against `expected_answer_summary` from the eval set, which
    was written from the source documents, rather than re-deriving the
    answer from the corpus. The judge's job is comparison, not retrieval.
  - Abstentions are never graded. Refusing is not a wrong answer, and
    feeding refusals to a correctness judge would bury the project's whole
    premise under a pile of INCORRECTs.

Grades are cached per (question, answer text), so re-running after adding
a rung costs only the new rows, and editing an answer invalidates just
that grade.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from refuses_to_lie.llm_client import call_groq

# "INCORRECT" contains "CORRECT", so the longer label has to be tested
# first or every wrong answer reads as right, silently halving the
# reported error rate -- the same trap the verifier's verdict parsing has.
_GRADE_CHECK_ORDER = ("INCORRECT", "PARTIAL", "CORRECT")
UNGRADED = "UNGRADED"

_GRADE_PROMPT = """You are grading a question-answering system against a \
reference answer written from the source documents.

Question: {question}

Reference answer: {reference}

System answer: {answer}

Does the system answer agree with the reference on the facts that matter -- \
figures, durations, eligibility and entitlements? Ignore differences in \
wording, length, formatting and citation markers. A system answer that is \
right but less detailed than the reference is CORRECT. One that states a \
different figure, duration or condition is INCORRECT. One that is right on \
some facts and wrong or missing on others is PARTIAL.

Reply with exactly one word: CORRECT, INCORRECT, or PARTIAL."""


@dataclass
class Grade:
    question_id: str
    config_id: str
    verdict: str
    answer: str

    @property
    def is_wrong(self) -> bool:
        """PARTIAL counts as wrong.

        A half-right answer about parental leave entitlement is still an
        answer somebody could act on and be wrong. Counting it as a pass
        would flatter the system in exactly the place this project cares
        about most.
        """
        return self.verdict in ("INCORRECT", "PARTIAL")


def _parse_grade(response: str) -> str:
    upper = response.upper()
    for verdict in _GRADE_CHECK_ORDER:
        if verdict in upper:
            return verdict
    return UNGRADED


def gradeable(row: dict) -> bool:
    """Only answers that claim to be answers, to questions with a reference.

    An abstention has no factual content to grade, and a question the
    corpus cannot answer has no reference to grade against -- the grid's
    own `error_floor` already covers that case.
    """
    return (
        "error" not in row
        and not row.get("abstained", False)
        and row.get("expected_answerable", False)
        and bool(row.get("expected_answer_summary"))
        and bool(row.get("answer"))
    )


def grade_cache_key(row: dict) -> str:
    """Keyed on the answer text, not the config.

    Two rungs that produced identical text get one grade and one API call,
    which on a ladder where later rungs often leave the answer untouched
    is most of them.
    """
    digest = hashlib.sha256(row["answer"].encode()).hexdigest()[:12]
    return f"grade-{row['question_id']}-{digest}"


def grade_row(row: dict, model: str) -> Grade:
    prompt = _GRADE_PROMPT.format(
        question=row["question"],
        reference=row["expected_answer_summary"],
        answer=row["answer"],
    )
    response = call_groq(prompt, model=model, temperature=0.0, cache_key=grade_cache_key(row))
    return Grade(
        question_id=row["question_id"],
        config_id=row["config_id"],
        verdict=_parse_grade(response),
        answer=row["answer"],
    )


def accuracy(grades: list[Grade]) -> tuple[float, float]:
    """(correct rate, wrong rate) over graded answers.

    They do not sum to 1: a grade the judge did not return cleanly is
    counted in neither, so a parsing failure cannot silently inflate
    either side.
    """
    scored = [g for g in grades if g.verdict != UNGRADED]
    if not scored:
        return 0.0, 0.0
    correct = sum(1 for g in scored if g.verdict == "CORRECT")
    wrong = sum(1 for g in scored if g.is_wrong)
    return correct / len(scored), wrong / len(scored)
