from refuses_to_lie.calibration import (
    Scored,
    best_threshold,
    separation,
    sweep,
)


def _s(composite: float, wrong: bool = False, abstained: bool = False, **components) -> Scored:
    base = {"retrieval": composite, "groundedness": composite, "agreement": composite}
    base.update(components)
    base["composite"] = composite
    return Scored("Q", composite, base, wrong, abstained)


def test_raising_the_threshold_lowers_coverage():
    scored = [_s(0.6), _s(0.7), _s(0.8), _s(0.9)]
    points = sweep(scored, applied_threshold=0.5, steps=3)
    coverages = [p.coverage for p in points]
    assert coverages == sorted(coverages, reverse=True)
    assert points[0].coverage == 1.0


def test_the_sweep_never_reports_below_the_threshold_that_ran():
    # Rows abstained under the applied threshold had their answer text
    # replaced, so there is nothing to score for a lower threshold.
    # Reporting one would be fabricating a measurement.
    points = sweep([_s(0.6), _s(0.9)], applied_threshold=0.5)
    assert all(p.threshold >= 0.5 for p in points)


def test_error_rate_is_measured_over_answers_actually_given():
    scored = [_s(0.6, wrong=True), _s(0.9, wrong=False)]
    points = sweep(scored, applied_threshold=0.5, steps=2)
    assert points[0].error_rate == 0.5  # both answered, one wrong
    assert points[-1].error_rate == 0.0  # only the 0.9 survives, and it is right


def test_abstained_rows_count_against_coverage_but_never_as_errors():
    # Refusing is the behaviour the project rewards.
    scored = [_s(0.9), _s(0.4, wrong=True, abstained=True)]
    point = sweep(scored, applied_threshold=0.5, steps=1)[0]
    assert point.coverage == 0.5
    assert point.error_rate == 0.0


def test_separation_detects_a_discriminating_signal():
    scored = [_s(0.9), _s(0.85), _s(0.3, wrong=True), _s(0.35, wrong=True)]
    gap = separation(scored)["composite"]["gap"]
    assert gap > 0.4


def test_separation_detects_a_useless_signal():
    # The finding that matters most: if right and wrong answers score the
    # same, no threshold can help and the signal needs rebuilding.
    scored = [_s(0.7), _s(0.7, wrong=True)]
    assert separation(scored)["composite"]["gap"] == 0.0


def test_best_threshold_picks_the_highest_coverage_within_budget():
    scored = [_s(0.6, wrong=True), _s(0.8), _s(0.9)]
    points = sweep(scored, applied_threshold=0.5, steps=5)
    pick = best_threshold(points, max_error=0.0)
    assert pick is not None
    assert pick.error_rate == 0.0
    assert pick.answered == 2  # keeps both correct answers, drops the wrong one


def test_best_threshold_returns_none_when_the_budget_is_unreachable():
    # A real answer, not a failure: refusing more cannot fix a system that
    # is wrong even when it is most confident.
    scored = [_s(0.9, wrong=True), _s(0.95, wrong=True)]
    points = sweep(scored, applied_threshold=0.5)
    assert best_threshold(points, max_error=0.1) is None


def test_sweep_of_nothing_is_empty_not_a_crash():
    assert sweep([], applied_threshold=0.5) == []
