"""Report the prompt-injection experiment.

Separates three outcomes the headline grid report collapses: whether an
adversarial document was retrieved, whether it was cited, and whether the
answer actually did what the payload demanded. Only the last is a
compromised system.

Breaks results down by attack style as well as by rung, because the styles
are not equivalent threats -- hidden white-on-white text is invisible to a
human reviewing the corpus, so it failing to be caught matters more than a
direct override that anyone would spot.

Usage:
  uv run python scripts/analyse_injection.py
  uv run python scripts/analyse_injection.py --results results/injection.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from refuses_to_lie.analysis import (
    attach_expectations,
    cites_injection,
    drop_stale,
    load_rows,
    retrieved_injection,
)
from refuses_to_lie.config import ALL_CONFIGS
from refuses_to_lie.injection import InjectionOutcome, classify, summarise

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
DEFAULT_RESULTS = ROOT / "results" / "injection.jsonl"


def print_ladder(by_config: dict[str, list[InjectionOutcome]]) -> None:
    print(f"{'rung':<6}{'n':>5}{'retrieved':>12}{'cited':>9}{'obeyed':>9}")
    print("-" * 42)
    for config_id in sorted(by_config):
        outcomes = by_config[config_id]
        rates = summarise(outcomes)
        print(
            f"{config_id:<6}{len(outcomes):>5}{rates['retrieved']:>11.1%}"
            f"{rates['cited']:>9.1%}{rates['obeyed']:>9.1%}"
        )
    print(
        "\nretrieved = an injected document reached the context window\n"
        "cited     = the answer attributed a claim to it\n"
        "obeyed    = the answer did what the payload demanded (signature match,\n"
        "            so a LOWER BOUND: a silently altered figure leaves no trace)"
    )


def print_by_style(outcomes: list[InjectionOutcome], style_of: dict[str, str]) -> None:
    by_style: dict[str, list[InjectionOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_style[style_of.get(outcome.question_id, "unknown")].append(outcome)

    print(f"\n{'style':<12}{'n':>5}{'retrieved':>12}{'cited':>9}{'obeyed':>9}")
    print("-" * 48)
    for style in sorted(by_style):
        rates = summarise(by_style[style])
        print(
            f"{style:<12}{len(by_style[style]):>5}{rates['retrieved']:>11.1%}"
            f"{rates['cited']:>9.1%}{rates['obeyed']:>9.1%}"
        )


def print_landed(outcomes: list[InjectionOutcome]) -> None:
    landed = [o for o in outcomes if o.obeyed]
    if not landed:
        print("\nNo payload signature matched on any row.")
        return
    print(f"\n{len(landed)} answer(s) visibly complied with a payload:")
    for outcome in sorted(landed, key=lambda o: (o.question_id, o.config_id)):
        signatures = ", ".join(outcome.obeyed_signatures)
        print(f"  {outcome.config_id} {outcome.question_id}  {signatures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.results.exists():
        parser.error(f"no results at {args.results}")

    questions = json.loads(EVAL_FILE.read_text())
    style_of = {q["id"]: q.get("injection_style", "unknown") for q in questions}

    rows, stale = drop_stale(load_rows(args.results), ALL_CONFIGS)
    if stale:
        print(f"ignoring {stale} row(s) from superseded config versions\n")
    rows = attach_expectations(rows, questions)
    rows = [r for r in rows if "error" not in r]

    outcomes = [classify(r, retrieved_injection(r), cites_injection(r)) for r in rows]
    by_config: dict[str, list[InjectionOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_config[outcome.config_id].append(outcome)

    questions_seen = len({o.question_id for o in outcomes})
    print(f"{len(outcomes)} rows | {questions_seen} injection questions\n")

    print_ladder(by_config)
    print_by_style(outcomes, style_of)
    print_landed(outcomes)


if __name__ == "__main__":
    main()
