"""Bookkeeping for a resumable eval grid.

The grid is thousands of LLM calls against rate-limited free tiers, so it
will be interrupted — by a 429, a crash, or someone pressing Ctrl-C. The
rules that make that survivable live here rather than in the CLI, because
getting them subtly wrong is the difference between resuming a run and
silently reusing results that no longer mean anything:

  - a row counts as done only if it SUCCEEDED, so a transient failure is
    retried rather than baked in as a permanent hole in the grid
  - every row carries a fingerprint of the config that produced it, so
    editing a model name or a top_k invalidates those rows automatically
    instead of mixing settings within one reported number
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

from refuses_to_lie.config import RunConfig

# Cosmetic: changing a label should not invalidate completed work.
NON_SUBSTANTIVE_FIELDS = frozenset({"label"})


def config_fingerprint(config: RunConfig) -> str:
    """Short hash over everything about a config that could change output."""
    substantive = {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.name not in NON_SUBSTANTIVE_FIELDS
    }
    blob = json.dumps(substantive, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def corpus_tag(includes_injected: bool) -> str:
    """Which corpus a row was produced against.

    Part of both keys below, because the corpus is an input to the answer
    exactly as much as the config is. Without it, a clean-baseline run
    would skip every row a contaminated run had already done, and worse,
    replay that run's cached generations -- answers written while looking
    at adversarial documents -- as though they were clean results.
    """
    return "inj" if includes_injected else "clean"


def cache_key(config: RunConfig, question_id: str, includes_injected: bool = True) -> str:
    """Stable per (config, corpus, question); invalidated by config changes."""
    fingerprint = config_fingerprint(config)
    return f"{config.id}-{fingerprint}-{corpus_tag(includes_injected)}-{question_id}"


Completion = tuple[str, str, str, str]


def load_completed(path: Path) -> set[Completion]:
    """(config_id, question_id, fingerprint, corpus) rows that succeeded.

    Failed rows stay in the file as a record of what went wrong but are not
    treated as done, so rerunning the same command retries exactly them.
    """
    if not path.exists():
        return set()
    done: set[Completion] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        done.add(
            (
                row["config_id"],
                row["question_id"],
                row["config_fingerprint"],
                corpus_tag(row.get("corpus_includes_injected", True)),
            )
        )
    return done


def pending_work(
    questions: list[dict],
    configs: list[RunConfig],
    completed: set[Completion],
    includes_injected: bool = True,
) -> list[tuple[dict, RunConfig]]:
    tag = corpus_tag(includes_injected)
    return [
        (question, config)
        for config in configs
        for question in questions
        if (config.id, question["id"], config_fingerprint(config), tag) not in completed
    ]


def stratified_sample(questions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Sample proportionally across categories.

    A validation slice is only useful if it exercises the hard paths, so
    taking a plain random sample risks a slice with no abstention cases or
    no injection cases in it at all.
    """
    by_category: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        by_category[question["category"]].append(question)

    rng = random.Random(seed)
    picked: list[dict] = []
    for _, group in sorted(by_category.items()):
        share = max(1, round(n * len(group) / len(questions)))
        picked.extend(rng.sample(group, min(share, len(group))))
    rng.shuffle(picked)
    return picked[:n]
