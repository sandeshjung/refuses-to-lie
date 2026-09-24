"""Grade the grid's answers for factual correctness.

Turns the grid's error FLOOR into a real error rate. The floor only counts
answers that were provably unwarranted; this counts the ones that were
confidently wrong -- the failure the project is named after.

Runs on Groq, deliberately: it is a different model from the generator, so
it is not grading its own homework.

Do NOT run this while a grid run is in flight. Groq is also the verifier's
provider, so rungs E and F spend the same per-minute budget, and a grading
sweep alongside them rate-limits the grid. Learned the hard way: 64
grading calls took out a rung E row mid-run.

Usage:
  uv run python scripts/grade_answers.py --results results/clean.jsonl
  uv run python scripts/grade_answers.py --configs A,F
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from refuses_to_lie.analysis import attach_expectations, drop_stale, load_rows
from refuses_to_lie.config import LADDER, A
from refuses_to_lie.grading import Grade, accuracy, grade_row, gradeable

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "eval" / "questions.json"
DEFAULT_RESULTS = ROOT / "results" / "clean.jsonl"


def load_existing(path: Path) -> dict[tuple[str, str], Grade]:
    """Grades already written, so a rerun costs nothing for them."""
    if not path.exists():
        return {}
    existing = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            grade = Grade(**row)
            existing[(grade.config_id, grade.question_id)] = grade
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--grades", type=Path, default=None)
    parser.add_argument("--configs", default="", help="comma-separated rung ids")
    parser.add_argument("--model", default=A.verifier_model)
    args = parser.parse_args()

    grades_path = args.grades or args.results.with_name(args.results.stem + "-grades.jsonl")

    rows, _ = drop_stale(load_rows(args.results), LADDER)
    rows = attach_expectations(rows, json.loads(EVAL_FILE.read_text()))
    if args.configs:
        wanted = {c.strip().upper() for c in args.configs.split(",")}
        rows = [r for r in rows if r["config_id"] in wanted]

    to_grade = [r for r in rows if gradeable(r)]
    existing = load_existing(grades_path)
    pending = [r for r in to_grade if (r["config_id"], r["question_id"]) not in existing]

    print(
        f"{len(rows)} rows | {len(to_grade)} gradeable "
        f"(answered, answerable, has reference)\n"
        f"{len(existing)} already graded, {len(pending)} to grade -> {grades_path}\n"
    )

    grades = list(existing.values())
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    with grades_path.open("a") as out:
        for i, row in enumerate(pending, start=1):
            grade = grade_row(row, args.model)
            out.write(json.dumps(grade.__dict__) + "\n")
            out.flush()
            grades.append(grade)
            if i % 10 == 0 or i == len(pending):
                print(
                    f"[{i}/{len(pending)}] {grade.config_id} {grade.question_id} "
                    f"{grade.verdict}",
                    flush=True,
                )

    by_config: dict[str, list[Grade]] = defaultdict(list)
    for grade in grades:
        by_config[grade.config_id].append(grade)

    print(f"\n{'rung':<6}{'graded':>8}{'correct':>10}{'wrong':>8}")
    print("-" * 32)
    for config_id in sorted(by_config):
        group = by_config[config_id]
        correct, wrong = accuracy(group)
        print(f"{config_id:<6}{len(group):>8}{correct:>9.1%}{wrong:>8.1%}")
    print(
        "\nwrong = INCORRECT or PARTIAL. A half-right answer about an "
        "entitlement\n        is still one somebody could act on and be wrong."
    )


if __name__ == "__main__":
    main()
