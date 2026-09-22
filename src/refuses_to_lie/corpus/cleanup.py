"""Stage 1: page cleanup.

Drop repeated headers/footers and table-of-contents pages before any
heading detection happens, so those don't need to be special-cased later.
"""

from __future__ import annotations

import re
from collections import Counter

_TOC_HEADER_LINES = {"table of contents", "contents"}
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_URL_PAGE_FOOTER_RE = re.compile(r"^https?://\S+\s+\d+/\d+$")


def strip_boilerplate(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop lines repeated across most pages (running headers/footers) and
    standalone page-number / print-footer lines."""
    counts: Counter[str] = Counter()
    for _, text in pages:
        for line in text.split("\n"):
            s = line.strip()
            if s:
                counts[s] += 1
    threshold = max(2, len(pages) // 2)
    boilerplate = {line for line, c in counts.items() if c >= threshold}

    cleaned = []
    for page_num, text in pages:
        kept = [
            line for line in text.split("\n") if not _is_noise_line(line.strip(), boilerplate)
        ]
        cleaned.append((page_num, "\n".join(kept)))
    return cleaned


def _is_noise_line(line: str, boilerplate: set[str]) -> bool:
    if not line:
        return False
    return bool(
        line in boilerplate or _PAGE_NUM_RE.match(line) or _URL_PAGE_FOOTER_RE.match(line)
    )


def is_toc_page(page_text: str) -> bool:
    """A table-of-contents page reuses real section numbers next to
    dot-leader page references, and some entries (short, all title-case,
    e.g. "4.2. General Public Holiday (Bank Holiday) Entitlement") pass the
    same heading-shape test as a real heading. Skip the whole page rather
    than rely on that test alone to reject every ToC line."""
    first_line = next((s.strip().lower() for s in page_text.split("\n") if s.strip()), "")
    return first_line in _TOC_HEADER_LINES
