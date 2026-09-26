"""Build and check the provenance register, and measure how forgeable the
naive alternative is.

Three checks, none of which call an LLM:

  1. coverage   every document the loader indexes is in the register, so
                rung H cannot silently lose a real policy
  2. integrity  no registered file is missing or altered since registration
  3. forgery    how often the "looks like a controlled document" rule would
                trust each attack style -- the measurement that decides
                whether cover-sheet trust is a defence or a decoration

Usage:
  uv run python scripts/check_provenance.py            # check only
  uv run python scripts/check_provenance.py --write    # (re)build the register first
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from refuses_to_lie.pipeline import load_corpus_chunks
from refuses_to_lie.provenance import (
    build_register,
    find_tampering,
    load_register,
    looks_controlled,
    save_register,
)

ROOT = Path(__file__).resolve().parent.parent
EMPLOYER = ROOT / "corpus" / "employer"
STATUTORY = ROOT / "corpus" / "statutory"
INJECTED = ROOT / "corpus" / "injected"
REGISTER = ROOT / "corpus" / "register.json"

STYLE_BY_PREFIX = {"INJ-D": "direct", "INJ-H": "hidden", "INJ-M": "metadata"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rebuild the register first")
    args = parser.parse_args()

    if args.write:
        register = build_register(EMPLOYER, STATUTORY)
        save_register(register, REGISTER)
        print(f"wrote {len(register)} documents -> {REGISTER.relative_to(ROOT)}\n")
    if not REGISTER.exists():
        parser.error("no register yet - run with --write")

    registered = set(load_register(REGISTER))
    indexed = {c.doc_id for c in load_corpus_chunks(EMPLOYER, STATUTORY)}
    missing = sorted(indexed - registered)
    covered = len(indexed & registered)
    print(f"1. coverage   {len(indexed)} indexed documents, {covered} registered")
    for doc_id in missing:
        print(f"     NOT REGISTERED: {doc_id}")

    problems = find_tampering(REGISTER, EMPLOYER, STATUTORY)
    print(f"2. integrity  {'ok' if not problems else f'{len(problems)} problem(s)'}")
    for problem in problems:
        print(f"     {problem}")

    print("3. forgery    would 'looks like a controlled document' trust it?")
    groups: dict[str, list[bool]] = defaultdict(list)
    for path in sorted(EMPLOYER.glob("*.pdf")):
        groups["legit: employer"].append(looks_controlled(path))
    for path in sorted(STATUTORY.glob("*.pdf")):
        groups["legit: statutory"].append(looks_controlled(path))
    fooled = []
    for path in sorted(INJECTED.glob("*.pdf")):
        style = STYLE_BY_PREFIX.get(path.stem[:5], "unknown")
        trusted = looks_controlled(path)
        groups[f"attack: {style}"].append(trusted)
        if trusted:
            fooled.append(path.stem)

    for name in sorted(groups):
        results = groups[name]
        print(f"     {name:<18}{sum(results):>3}/{len(results):<3} trusted")
    for stem in fooled:
        print(f"     forged cover sheet accepted: {stem}")
    print(
        "\n   The register trusts by identity and content hash, so none of the\n"
        "   injected documents pass it -- by construction, since they were never\n"
        "   registered. Section 3 is the part that is actually an experiment."
    )


if __name__ == "__main__":
    main()
