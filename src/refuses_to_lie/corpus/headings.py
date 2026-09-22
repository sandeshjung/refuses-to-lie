"""Stage 2: line classification.

Decide whether a line is a numbered heading, an unnumbered heading, or body
text.
"""

from __future__ import annotations

import re

MAX_NUMBERED_HEADING_WORDS = 10
MAX_UNNUMBERED_HEADING_WORDS = 8
MAX_PLAIN_ENGLISH_HEADING_WORDS = 6

_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "in",
    "on",
    "for",
    "with",
    "to",
    "at",
    "by",
    "or",
    "from",
    "&",
}


def strip_toc_tail(text: str) -> str:
    """Drop a trailing dot-leader + page number, e.g. '...Purpose .... 4'."""
    return re.sub(r"\.{2,}.*$", "", text).strip()


def is_heading_shaped(remainder: str, max_words: int) -> bool:
    """Heading text in this corpus is short and every content word is
    capitalized (Title Case or ALL CAPS) — unlike ordinary sentences, which
    are full of lowercase function/content words. This is what tells a real
    heading ("4.1 PART I") apart from a numbered body clause that happens to
    share the same "N.M " shape ("4.1 The framework comprises five parts:")."""
    remainder = strip_toc_tail(remainder)
    words = remainder.split()
    if not words or len(words) > max_words:
        return False
    if remainder.rstrip().endswith((".", ",", ";")):
        return False
    for word in words:
        core = word.strip("():,.;’'\"")
        if not core:
            continue
        if core.lower() in _STOPWORDS:
            continue
        if core.isdigit():
            # A bare number ("1", "2") carries no case at all, so it can't
            # fail or pass the capitalization test on its own merits — skip
            # it, same as a stopword. Real headings like "Class 1 National
            # Insurance thresholds" contain these; rejecting the whole line
            # over one digit token was silently dropping every heading with
            # a number in its title, not just numbered-list prefixes.
            continue
        if not (core[0].isupper() or core.isupper()):
            return False
    return True


def is_plain_english_heading_shaped(remainder: str, max_words: int) -> bool:
    """A looser heading-shape test for documents that never use dotted (X.Y)
    sub-numbering — GOV.UK-style guidance, whose top-level headings are
    natural-language phrases ("What you'll get", "How to claim") rather than
    Title Case or ALL CAPS, so is_heading_shaped's per-word capitalization
    check rejects them. NHS policy docs always use dotted numbering
    somewhere for their X.Y sub-sections, so this path never runs for them —
    see document_uses_dotted_numbering."""
    remainder = strip_toc_tail(remainder)
    words = remainder.split()
    if not words or len(words) > max_words:
        return False
    if not remainder[0].isupper():
        return False
    return not remainder.rstrip().endswith((".", ",", ";", ":", "?", "!"))


def document_uses_dotted_numbering(pages: list[tuple[int, str]]) -> bool:
    """True if any numbered line in the document has a dotted section
    number (e.g. "10.1", "4.2") — the signal that distinguishes NHS-style
    numbered-outline documents (which always have these for their
    sub-sections) from GOV.UK-style guidance (which only ever numbers a
    flat top-level list, e.g. "1.", "2.", "3.")."""
    for _, text in pages:
        for line in text.split("\n"):
            m = _NUM_RE.match(line.strip())
            if m and "." in m.group(1):
                return True
    return False


def match_numbered_heading(
    line: str, allow_plain_english: bool = False
) -> tuple[str, str] | None:
    """Return (section_no, heading_text) if `line` is a numbered heading."""
    m = _NUM_RE.match(line)
    if not m:
        return None
    no, remainder = m.groups()
    if is_heading_shaped(remainder, MAX_NUMBERED_HEADING_WORDS):
        return no, strip_toc_tail(remainder)
    if (
        allow_plain_english
        and "." not in no
        and is_plain_english_heading_shaped(remainder, MAX_PLAIN_ENGLISH_HEADING_WORDS)
    ):
        return no, strip_toc_tail(remainder)
    return None


def match_unnumbered_heading(line: str) -> str | None:
    """Return the heading text if `line` is an unnumbered bold/capitalised
    sub-heading (only meaningful once we're already inside a section — see
    SectionTree.add_line)."""
    if is_heading_shaped(line, MAX_UNNUMBERED_HEADING_WORDS):
        return line
    return None


def section_level(section_no: str) -> int:
    """'10.0' -> level 1 (top-level, X.0 is this corpus's "no sub-part"
    marker), '10.1' -> level 2, '1' -> level 1."""
    parts = section_no.split(".")
    if len(parts) > 1 and parts[-1] == "0":
        parts = parts[:-1]
    return len(parts)
