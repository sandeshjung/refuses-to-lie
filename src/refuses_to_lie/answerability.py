"""Should this question be answered from this context at all?

Replaces the composite confidence score for rung G. The calibration run
measured the composite as anti-correlated with correctness -- wrong answers
scored HIGHER -- and each component failed for an identifiable reason:

  - agreement rewarded confident fabrication: a model that invents a
    figure invents the same one three times
  - groundedness saturated at 1.0 on wrong answers, because they were
    faithfully grounded in text that did not answer the question
  - retrieval margin was the one weak positive, and it flipped strongly
    negative under injection, where adversarial pages retrieve best

What all three miss is the question. Each scores the answer against the
context; none asks whether the context contains what was asked. That is
exactly the near-miss failure -- the right policy retrieved, the specific
figure absent -- which produced most unwarranted answers. So this asks it
directly, before an answer is written, and a "no" costs no generation.

Two deliberate choices in the prompt:
  - a question whose assumption the excerpts CONTRADICT is answerable,
    because the right response to a false premise is to correct it, and
    the eval's own notes say so
  - "the excerpts defer the detail to another document" is PARTIAL, not
    ANSWERABLE, because this corpus repeatedly defers its numbers to
    policies it does not contain
"""

from __future__ import annotations

from dataclasses import dataclass

from refuses_to_lie.config import RunConfig
from refuses_to_lie.generation import format_excerpts
from refuses_to_lie.llm_client import call_groq
from refuses_to_lie.retrieval import Hit

# "UNANSWERABLE" contains "ANSWERABLE": check the longer label first or
# every refusal parses as a green light -- the same trap as the verifier's
# UNSUPPORTED/SUPPORTED and the judge's INCORRECT/CORRECT.
_VERDICT_CHECK_ORDER = ("UNANSWERABLE", "PARTIAL", "ANSWERABLE")

SCORES = {"ANSWERABLE": 1.0, "PARTIAL": 0.5, "UNANSWERABLE": 0.0}
UNKNOWN = "UNKNOWN"

_PROMPT = """You are checking whether policy excerpts can support an answer, \
before any answer is written.

Question: {question}

Excerpts:
{excerpts}

Do the excerpts contain the specific information needed to answer this question \
correctly?

- ANSWERABLE: the excerpts state the specific fact, figure, rule or entitlement asked \
for; OR they directly contradict an assumption in the question, so it can be answered \
by correcting that assumption.
- PARTIAL: the excerpts cover the topic but not the specific point asked -- for \
example a related entitlement, a different group of staff, a different year, or they \
refer the reader to another document for the detail.
- UNANSWERABLE: the excerpts do not address the question.

Reply with exactly one word: ANSWERABLE, PARTIAL, or UNANSWERABLE."""


@dataclass
class AnswerabilitySignal:
    verdict: str
    answerability: float

    @property
    def composite(self) -> float:
        """Named to match ConfidenceSignal, so thresholding and reporting
        treat both signals identically."""
        return self.answerability


def parse_verdict(response: str) -> str:
    upper = response.upper()
    for verdict in _VERDICT_CHECK_ORDER:
        if verdict in upper:
            return verdict
    return UNKNOWN


def check_answerability(
    question: str, hits: list[Hit], config: RunConfig, cache_key: str | None = None
) -> AnswerabilitySignal:
    """Judge the context against the question, with the verifier's model.

    An unparseable reply scores 0.0: when the gate cannot tell, the system
    refuses. Failing open would make a malformed judge response the one
    input guaranteed to be answered.
    """
    prompt = _PROMPT.format(question=question, excerpts=format_excerpts(hits))
    response = call_groq(
        prompt, model=config.verifier_model, temperature=0.0, cache_key=cache_key
    )
    verdict = parse_verdict(response)
    return AnswerabilitySignal(verdict=verdict, answerability=SCORES.get(verdict, 0.0))
