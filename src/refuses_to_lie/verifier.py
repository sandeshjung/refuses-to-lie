"""Groundedness verification: for each cited claim in a generated answer,
check whether the excerpt it cites actually supports it.

Uses a different model (Groq) from the generator (Gemini) to answer a
focused SUPPORTED/UNSUPPORTED/PARTIAL question per (claim, cited excerpt)
pair — checking the generator's own citations against the source text it
says backs them, not re-answering the question from scratch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from refuses_to_lie.config import RunConfig
from refuses_to_lie.generation import GeneratedAnswer
from refuses_to_lie.llm_client import call_groq
from refuses_to_lie.retrieval import Hit

# One claim = the shortest run of text up to and including a citation
# marker, ending at the next sentence boundary. Matches real generated
# output, including markdown bullets, where a citation sits at the end of
# its sentence — verified against real Gemini output, not assumed.
_CLAIM_RE = re.compile(r"([^.\n]*\[([\w.\-]+-\d{4})\][^.\n]*\.)")

# Longer/more specific labels first: "SUPPORTED" is a substring of
# "UNSUPPORTED", so scanning in the other order would match "SUPPORTED"
# against an "UNSUPPORTED" response and get every unsupported claim backwards.
_VERDICT_CHECK_ORDER = ("UNSUPPORTED", "PARTIAL", "SUPPORTED")

_VERIFY_PROMPT = """Source excerpt:
{source}

Claim: "{claim}"

Does the source excerpt support this claim? Reply with exactly one word: \
SUPPORTED, UNSUPPORTED, or PARTIAL."""


@dataclass
class ClaimVerdict:
    claim_text: str
    chunk_id: str
    # SUPPORTED, UNSUPPORTED, PARTIAL, or UNKNOWN (cited chunk_id wasn't in context)
    verdict: str


@dataclass
class VerifiedAnswer:
    answer: GeneratedAnswer
    claim_verdicts: list[ClaimVerdict]

    @property
    def groundedness(self) -> float:
        """Fraction of claims marked SUPPORTED. 1.0 for an answer with no
        citation-bearing claims at all (nothing to contradict)."""
        if not self.claim_verdicts:
            return 1.0
        supported = sum(1 for v in self.claim_verdicts if v.verdict == "SUPPORTED")
        return supported / len(self.claim_verdicts)

    @property
    def all_supported(self) -> bool:
        return all(v.verdict == "SUPPORTED" for v in self.claim_verdicts)


def _split_claims(text: str) -> list[tuple[str, str]]:
    """Return (claim_sentence, chunk_id) pairs, one per cited sentence."""
    return [(m.group(1).strip(), m.group(2)) for m in _CLAIM_RE.finditer(text)]


def _parse_verdict(response: str) -> str:
    upper = response.strip().upper()
    for label in _VERDICT_CHECK_ORDER:
        if label in upper:
            return label
    return "UNKNOWN"


def verify_answer(
    answer: GeneratedAnswer,
    hits: list[Hit],
    config: RunConfig,
    cache_key_prefix: str | None = None,
) -> VerifiedAnswer:
    hits_by_id = {h.chunk.chunk_id: h for h in hits}
    verdicts: list[ClaimVerdict] = []

    for i, (claim_text, chunk_id) in enumerate(_split_claims(answer.text)):
        hit = hits_by_id.get(chunk_id)
        if hit is None:
            verdicts.append(ClaimVerdict(claim_text, chunk_id, "UNKNOWN"))
            continue

        prompt = _VERIFY_PROMPT.format(source=hit.chunk.text, claim=claim_text)
        cache_key = f"{cache_key_prefix}-claim{i}" if cache_key_prefix else None
        response = call_groq(
            prompt, model=config.verifier_model, temperature=0.0, cache_key=cache_key
        )
        verdicts.append(ClaimVerdict(claim_text, chunk_id, _parse_verdict(response)))

    return VerifiedAnswer(answer=answer, claim_verdicts=verdicts)


def drop_unsupported_claims(verified: VerifiedAnswer) -> str:
    """Rung E's "drop_unsupported" verifier action: strip any sentence
    whose cited claim wasn't SUPPORTED, leaving the rest of the answer
    intact."""
    text = verified.answer.text
    for v in verified.claim_verdicts:
        if v.verdict != "SUPPORTED":
            text = text.replace(v.claim_text, "").strip()
    return text
