"""Composite abstention signal: retrieval confidence + groundedness +
cross-sample agreement, combined into one score to threshold against.

None of the three inputs is trusted alone — asking a model how confident it
is doesn't work, so every input here is built from externally observable
behavior instead: how much the top retrieval result stands out from the
rest, whether the generator's own citations survive an independent
groundedness check (verifier.py), and whether resampling the generator at
its configured temperature keeps landing on the same citations.

The weights below are a starting point, not a calibrated answer — actual
calibration (finding a threshold that corresponds to a real error rate)
happens once the eval set exists and is a separate step, not guessed here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from refuses_to_lie.config import RunConfig
from refuses_to_lie.generation import GeneratedAnswer, generate_answer
from refuses_to_lie.retrieval import Hit
from refuses_to_lie.verifier import VerifiedAnswer

DEFAULT_WEIGHTS: dict[str, float] = {
    "retrieval": 0.3,
    "groundedness": 0.4,
    "agreement": 0.3,
}


def retrieval_confidence(hits: list[Hit]) -> float:
    """How much the top retrieved chunk stands out from the rest of the
    returned set, on a query-relative [0, 1] scale.

    Margin-based rather than absolute-score-based on purpose: dense hits
    are bounded cosine similarities (~0-1) but hybrid hits are RRF-fused
    scores on a different, unbounded scale, so comparing raw magnitudes
    across retrieval modes isn't meaningful. The relative gap between the
    best and second-best result, scaled against the full spread of scores
    actually returned for this query, is comparable either way.
    """
    if not hits:
        return 0.0
    if len(hits) == 1:
        return 1.0
    scores = [h.score for h in hits]
    top, second, worst = scores[0], scores[1], scores[-1]
    spread = top - worst
    if spread <= 0:
        return 0.5  # every candidate scored identically — no signal either way
    return (top - second) / spread


def _pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 1.0
    scores = []
    for a, b in itertools.combinations(sets, 2):
        if not a and not b:
            scores.append(1.0)  # neither sample cited anything — nothing to disagree on
        elif not a or not b:
            scores.append(0.0)
        else:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores)


@dataclass
class AgreementResult:
    samples: list[GeneratedAnswer]
    citation_overlap: float
    any_abstained: bool
    all_abstained: bool

    @property
    def score(self) -> float:
        """1.0 = samples fully agree (consistent citations, or consistent
        abstention); 0.0 = maximal disagreement (one sample abstained, a
        differently-sampled run of the same question didn't)."""
        if self.all_abstained:
            return 1.0
        if self.any_abstained:
            return 0.0
        return self.citation_overlap


def sample_agreement(
    question: str,
    hits: list[Hit],
    config: RunConfig,
    cache_key_prefix: str | None = None,
) -> AgreementResult:
    """Generate the answer config.agreement_samples times (at config's own
    temperature — no hidden extra knob outside RunConfig) and measure how
    much the resulting citations agree.

    Requires config.require_citations=True: agreement is measured as
    citation-set overlap, which is only meaningful when citations exist.
    This holds for every ladder rung that actually uses abstention (F
    inherits require_citations=True from D via the ladder's replace()
    chain), so it's a real constraint, not a hypothetical one.
    """
    if not config.require_citations:
        raise ValueError("sample_agreement needs config.require_citations=True")

    samples = [
        generate_answer(
            question,
            hits,
            config,
            cache_key=(f"{cache_key_prefix}-sample{i}" if cache_key_prefix else None),
        )
        for i in range(config.agreement_samples)
    ]
    any_abstained = any(s.abstained for s in samples)
    all_abstained = all(s.abstained for s in samples)
    citation_sets = [{c.chunk_id for c in s.citations} for s in samples]

    return AgreementResult(
        samples=samples,
        citation_overlap=_pairwise_jaccard(citation_sets),
        any_abstained=any_abstained,
        all_abstained=all_abstained,
    )


@dataclass
class ConfidenceSignal:
    retrieval: float
    groundedness: float
    agreement: float
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def composite(self) -> float:
        return (
            self.weights["retrieval"] * self.retrieval
            + self.weights["groundedness"] * self.groundedness
            + self.weights["agreement"] * self.agreement
        )


def compute_confidence(
    hits: list[Hit],
    verified: VerifiedAnswer,
    agreement: AgreementResult,
    weights: dict[str, float] | None = None,
) -> ConfidenceSignal:
    return ConfidenceSignal(
        retrieval=retrieval_confidence(hits),
        groundedness=verified.groundedness,
        agreement=agreement.score,
        weights=dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS),
    )


def should_abstain(signal: ConfidenceSignal, threshold: float) -> bool:
    return signal.composite < threshold
