"""Turning the confidence score into a threshold that means something.

`abstain_threshold` shipped as 0.5 with a comment admitting it was a guess.
This module replaces the guess with a measurement: sweep the threshold over
the scores the grid actually produced and report what each one would have
cost in coverage and bought in correctness.

One hard limit, enforced rather than papered over. The grid ran WITH a
threshold applied, and an abstained row's answer text was replaced by the
refusal phrase. So for any threshold below the one that ran, the rows that
would newly be answered have no answer to score -- the data is censored,
not merely sparse. Sweeping downward would invent numbers. `sweep()`
therefore refuses to report below the applied threshold, and says why.

The second question this answers is more fundamental than where to put the
line: whether the signal separates right answers from wrong ones at all. A
threshold cannot rescue a score that does not discriminate, and
`separation()` measures that directly.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# Rung F records the first three; rung G records "answerability". Both
# record "composite", which is what the threshold is applied to.
COMPONENTS = ("retrieval", "groundedness", "agreement", "answerability", "composite")


@dataclass
class Scored:
    """One answered row with its confidence and whether it was wrong."""

    question_id: str
    composite: float
    components: dict[str, float]
    wrong: bool
    abstained: bool


@dataclass
class CalibrationPoint:
    threshold: float
    coverage: float
    error_rate: float
    answered: int
    total: int

    @property
    def summary(self) -> str:
        return (
            f"t={self.threshold:.2f}  coverage={self.coverage:6.1%}  "
            f"error={self.error_rate:6.1%}  ({self.answered}/{self.total})"
        )


def sweep(
    scored: list[Scored], applied_threshold: float, steps: int = 11
) -> list[CalibrationPoint]:
    """Coverage and error for each threshold at or above the one that ran.

    Raising the threshold can only turn answers into abstentions, and those
    answers were recorded, so every point here is measured rather than
    modelled.
    """
    if not scored:
        return []

    top = max(s.composite for s in scored)
    points = []
    for i in range(steps):
        threshold = applied_threshold + (top - applied_threshold) * i / max(steps - 1, 1)
        answered = [s for s in scored if not s.abstained and s.composite >= threshold]
        wrong = sum(1 for s in answered if s.wrong)
        points.append(
            CalibrationPoint(
                threshold=threshold,
                coverage=len(answered) / len(scored),
                error_rate=wrong / len(answered) if answered else 0.0,
                answered=len(answered),
                total=len(scored),
            )
        )
    return points


def separation(scored: list[Scored]) -> dict[str, dict[str, float]]:
    """Per component, the mean score for right vs wrong answers.

    If these are the same, no threshold on that component can help: the
    score is not carrying information about correctness, and the honest
    conclusion is that the signal needs rebuilding rather than retuning.
    """
    answered = [s for s in scored if not s.abstained]
    right = [s for s in answered if not s.wrong]
    wrong = [s for s in answered if s.wrong]

    result = {}
    for component in COMPONENTS:
        r = [s.components[component] for s in right if component in s.components]
        w = [s.components[component] for s in wrong if component in s.components]
        if not r and not w:
            continue  # this rung does not compute the component; omit, don't print zeros
        result[component] = {
            "right_mean": statistics.fmean(r) if r else 0.0,
            "wrong_mean": statistics.fmean(w) if w else 0.0,
            "gap": (statistics.fmean(r) if r else 0.0) - (statistics.fmean(w) if w else 0.0),
            "n_right": len(r),
            "n_wrong": len(w),
        }
    return result


def best_threshold(
    points: list[CalibrationPoint], max_error: float
) -> CalibrationPoint | None:
    """Highest coverage whose error rate stays within budget.

    Returns None when no threshold achieves it, which is a real answer:
    it means the system cannot hit that error rate by refusing more, only
    by being better.
    """
    viable = [p for p in points if p.error_rate <= max_error and p.answered > 0]
    return max(viable, key=lambda p: p.coverage) if viable else None
