"""Report what the eval grid measured.

Deterministic and free to re-run, so it is safe to point at a grid that is
still being written: it scores whatever rows exist. That makes it the right
thing to run against a validation slice before committing quota to the full
grid.

Usage:
  uv run python scripts/analyse_grid.py
  uv run python scripts/analyse_grid.py --results results/grid.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from refuses_to_lie.analysis import (
    ConfigScore,
    attach_expectations,
    coverage_error_points,
    drop_stale,
    load_rows,
    score_grid,
)
from refuses_to_lie.config import LADDER

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "grid.jsonl"
EVAL_FILE = ROOT / "eval" / "questions.json"

LABELS = {c.id: c.label for c in LADDER}


def _injection_cell(score: ConfigScore) -> str:
    """Blank out a number the rung cannot produce.

    A rung that never cites cannot be caught citing an injected document,
    so printing 0.0% there would read as "resisted the attack" when it
    actually means "not measurable".
    """
    return f"{score.injection_citation_rate:>7.1%}" if score.citation_rate else "    n/a"


def print_ladder(scores: list[ConfigScore]) -> None:
    print(
        f"{'rung':<5}{'n':>5}{'err':>5}{'cover':>8}{'floor':>8}{'f-abst':>8}{'f-ans':>8}{'inj':>8}"
    )
    print("-" * 55)
    for s in scores:
        print(
            f"{s.config_id:<5}{s.rows:>5}{s.errors:>5}{s.coverage:>7.1%}"
            f"{s.error_floor:>8.1%}{s.false_abstention_rate:>8.1%}"
            f"{s.false_answer_rate:>8.1%}{_injection_cell(s):>8}"
        )
    print(
        "\ncover  = questions answered rather than abstained\n"
        "floor  = of those answers, the share to questions the corpus cannot\n"
        "         answer. The ONLY rung-to-rung comparable error signal, and a\n"
        "         floor: wrong figures on answerable questions are not counted\n"
        "f-abst = answerable questions refused\n"
        "f-ans  = unanswerable questions answered anyway\n"
        "inj    = of those answers, the share citing an injected document.\n"
        "         n/a where the rung emits no citations, so it cannot be seen.\n"
        "         Do NOT add this to floor and compare down the ladder."
    )


def print_detail(scores: list[ConfigScore]) -> None:
    print(
        f"\n{'rung':<5}{'retr@k':>8}{'cited':>8}{'inj-ret':>9}{'inj-cite':>10}{'median s':>10}"
    )
    print("-" * 50)
    for s in scores:
        print(
            f"{s.config_id:<5}{s.retrieval_hit_rate:>7.1%}{s.citation_rate:>8.1%}"
            f"{s.injection_retrieved_rate:>9.1%}{s.injection_cited_rate:>10.1%}"
            f"{s.median_latency_s:>10.1f}"
        )
    print(
        "\ninj-ret  = ANY question where an injected document reached the context\n"
        "inj-cite = ...and was cited in the answer (retrieval leak vs landed attack)"
    )


def print_verdicts(scores: list[ConfigScore]) -> None:
    scored = [s for s in scores if s.verdicts]
    if not scored:
        return
    print("\nverifier verdicts (rungs with the verifier on)")
    print("-" * 50)
    for s in scored:
        total = sum(s.verdicts.values())
        breakdown = "  ".join(
            f"{verdict.lower()}={count / total:.1%}"
            for verdict, count in s.verdicts.most_common()
        )
        print(f"{s.config_id:<5}{total:>5} claims   {breakdown}")


def print_curve(scores: list[ConfigScore]) -> None:
    print("\ncoverage / error floor (comparable signal only)")
    print("-" * 50)
    for config_id, coverage, floor in coverage_error_points(scores):
        print(
            f"{config_id}  coverage={coverage:>6.1%}  "
            f"error_floor={floor:>6.1%}   {LABELS.get(config_id, '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.results.exists():
        parser.error(f"no results at {args.results} - run scripts/run_grid.py first")

    rows, stale = drop_stale(load_rows(args.results), LADDER)
    if stale:
        print(f"ignoring {stale} row(s) from superseded config versions\n")
    if not rows:
        parser.error("no rows match the current ladder - rerun scripts/run_grid.py")

    rows = attach_expectations(rows, json.loads(EVAL_FILE.read_text()))
    scores = score_grid(rows)
    questions = len({r["question_id"] for r in rows})
    print(f"{len(rows)} rows | {questions} questions | {len(scores)} configs\n")

    print_ladder(scores)
    print_detail(scores)
    print_verdicts(scores)
    print_curve(scores)


if __name__ == "__main__":
    main()
