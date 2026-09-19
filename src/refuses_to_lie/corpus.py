from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

MAX_NUMBERED_HEADING_WORDS = 10
MAX_UNNUMBERED_HEADING_WORDS = 8
MAX_PLAIN_ENGLISH_HEADING_WORDS = 6

_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "with", "to", "at",
    "by", "or", "from", "&",
}
_TOC_HEADER_LINES = {"table of contents", "contents"}
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_URL_PAGE_FOOTER_RE = re.compile(r"^https?://\S+\s+\d+/\d+$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading_trail: tuple[str, ...]
    section_no: str | None
    page_start: int
    page_end: int
    token_count: int

    @property
    def cite_label(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        if self.section_no:
            return f"{self.doc_id} §{self.section_no} ({pages})"
        return f"{self.doc_id} ({pages})"


# --------------------------------------------------------------------------
# Stage 1: page cleanup — drop repeated headers/footers and ToC pages before
# any heading detection happens, so those don't need to be special-cased
# later.
# --------------------------------------------------------------------------


def _strip_boilerplate(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
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
            line
            for line in text.split("\n")
            if not _is_noise_line(line.strip(), boilerplate)
        ]
        cleaned.append((page_num, "\n".join(kept)))
    return cleaned


def _is_noise_line(line: str, boilerplate: set[str]) -> bool:
    if not line:
        return False
    return bool(
        line in boilerplate
        or _PAGE_NUM_RE.match(line)
        or _URL_PAGE_FOOTER_RE.match(line)
    )


def _is_toc_page(page_text: str) -> bool:
    """A table-of-contents page reuses real section numbers next to
    dot-leader page references, and some entries (short, all title-case,
    """
    first_line = next(
        (s.strip().lower() for s in page_text.split("\n") if s.strip()), ""
    )
    return first_line in _TOC_HEADER_LINES


# --------------------------------------------------------------------------
# Stage 2: line classification — decide whether a line is a numbered
# heading, an unnumbered heading, or body text.
# --------------------------------------------------------------------------


def _strip_toc_tail(text: str) -> str:
    """Drop a trailing dot-leader + page number, e.g. '...Purpose .... 4'."""
    return re.sub(r"\.{2,}.*$", "", text).strip()


def _is_heading_shaped(remainder: str, max_words: int) -> bool:
    remainder = _strip_toc_tail(remainder)
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
        if not (core[0].isupper() or core.isupper()):
            return False
    return True


def _is_plain_english_heading_shaped(remainder: str, max_words: int) -> bool:
    remainder = _strip_toc_tail(remainder)
    words = remainder.split()
    if not words or len(words) > max_words:
        return False
    if not remainder[0].isupper():
        return False
    if remainder.rstrip().endswith((".", ",", ";", ":", "?", "!")):
        return False
    return True


def _document_uses_dotted_numbering(pages: list[tuple[int, str]]) -> bool:
    for _, text in pages:
        for line in text.split("\n"):
            m = _NUM_RE.match(line.strip())
            if m and "." in m.group(1):
                return True
    return False


def _match_numbered_heading(
    line: str, allow_plain_english: bool = False
) -> tuple[str, str] | None:
    """Return (section_no, heading_text) if `line` is a numbered heading."""
    m = _NUM_RE.match(line)
    if not m:
        return None
    no, remainder = m.groups()
    if _is_heading_shaped(remainder, MAX_NUMBERED_HEADING_WORDS):
        return no, _strip_toc_tail(remainder)
    if (
        allow_plain_english
        and "." not in no
        and _is_plain_english_heading_shaped(remainder, MAX_PLAIN_ENGLISH_HEADING_WORDS)
    ):
        return no, _strip_toc_tail(remainder)
    return None


def _match_unnumbered_heading(line: str) -> str | None:
    if _is_heading_shaped(line, MAX_UNNUMBERED_HEADING_WORDS):
        return line
    return None


def _section_level(section_no: str) -> int:
    parts = section_no.split(".")
    if len(parts) > 1 and parts[-1] == "0":
        parts = parts[:-1]
    return len(parts)


def _count_tokens(text: str) -> int:
    """Whitespace word count — a cheap proxy, not a real LLM tokenizer."""
    return len(text.split())


# --------------------------------------------------------------------------
# Stage 3: build the section tree — a stack of currently-open sections,
# where pushing a heading at level L closes any open sections at level >= L
# first (so a chunk's heading_trail is always its true root-to-leaf
# ancestry).
# --------------------------------------------------------------------------


@dataclass
class _OpenSection:
    level: int
    section_no: str | None
    heading_trail: tuple[str, ...]
    page_start: int
    lines: list[str] = field(default_factory=list)
    last_page: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.lines


class _SectionTree:
    """Consumes a document's lines in order and emits one Chunk per
    non-empty section, correctly nested by heading level."""

    def __init__(self, doc_id: str, allow_plain_english: bool = False):
        self.doc_id = doc_id
        self._allow_plain_english = allow_plain_english
        self._stack: list[_OpenSection] = []
        self._chunks: list[Chunk] = []

    def add_line(self, line: str, page_num: int) -> None:
        if self._stack:
            self._stack[-1].last_page = page_num
        if not line:
            return

        numbered = _match_numbered_heading(line, self._allow_plain_english)
        if numbered:
            section_no, heading = numbered
            self._open(_section_level(section_no), heading, section_no, page_num)
            return

        if self._stack:
            heading = _match_unnumbered_heading(line)
            if heading:
                nearest_numbered_level = self._nearest_numbered_level()
                self._open(nearest_numbered_level + 1, heading, None, page_num)
                return

        if self._stack:
            self._stack[-1].lines.append(line)
        # else: content before the first heading has no section to attach
        # to and is dropped (front matter, ToC leftovers, values statements).

    def finish(self) -> list[Chunk]:
        while self._stack:
            self._close_top()
        self._chunks.sort(key=lambda c: c.page_start)
        return [
            Chunk(
                chunk_id=f"{self.doc_id}-{i:04d}",
                doc_id=c.doc_id,
                text=c.text,
                heading_trail=c.heading_trail,
                section_no=c.section_no,
                page_start=c.page_start,
                page_end=c.page_end,
                token_count=c.token_count,
            )
            for i, c in enumerate(self._chunks, start=1)
        ]

    def _nearest_numbered_level(self) -> int:
        for entry in reversed(self._stack):
            if entry.section_no is not None:
                return entry.level
        return 0

    def _open(
        self, level: int, heading: str, section_no: str | None, page_num: int
    ) -> None:
        if self._should_fold_into_open_parent(level, section_no):
            self._fold_into_top(heading, page_num)
            return
        while self._stack and self._stack[-1].level >= level:
            self._close_top()
        parent_trail = self._stack[-1].heading_trail if self._stack else ()
        self._stack.append(
            _OpenSection(
                level=level,
                section_no=section_no,
                heading_trail=parent_trail + (heading,),
                page_start=page_num,
                last_page=page_num,
            )
        )

    def _should_fold_into_open_parent(self, level: int, section_no: str | None) -> bool:
        return (
            section_no is None
            and bool(self._stack)
            and self._stack[-1].is_empty
            and self._stack[-1].level < level
        )

    def _fold_into_top(self, heading: str, page_num: int) -> None:
        top = self._stack[-1]
        top.heading_trail = top.heading_trail[:-1] + (
            f"{top.heading_trail[-1]}: {heading}",
        )
        top.last_page = page_num

    def _close_top(self) -> None:
        sec = self._stack.pop()
        text = "\n".join(sec.lines).strip()
        if not text:
            return
        self._chunks.append(
            Chunk(
                chunk_id="",
                doc_id=self.doc_id,
                text=text,
                heading_trail=sec.heading_trail,
                section_no=sec.section_no,
                page_start=sec.page_start,
                page_end=sec.last_page or sec.page_start,
                token_count=_count_tokens(text),
            )
        )


def chunk_document(
    doc_id: str,
    pages: list[tuple[int, str]],
    min_tokens: int = 0,
) -> list[Chunk]:
    pages = _strip_boilerplate(pages)
    allow_plain_english = not _document_uses_dotted_numbering(pages)

    tree = _SectionTree(doc_id, allow_plain_english=allow_plain_english)
    for page_num, text in pages:
        if _is_toc_page(text):
            continue
        for raw_line in text.split("\n"):
            tree.add_line(raw_line.strip(), page_num)

    chunks = tree.finish()
    if min_tokens > 0:
        chunks = _merge_short_chunks(chunks, min_tokens, doc_id)
    return chunks


# --------------------------------------------------------------------------
# Optional post-processing and inspection helpers.
# --------------------------------------------------------------------------


def _merge_short_chunks(chunks: list[Chunk], min_tokens: int, doc_id: str) -> list[Chunk]:
    """Merge any chunk below min_tokens into its next neighbor (or, for a
    trailing short chunk, into its previous neighbor). Off by default —
    see chunk_document's docstring for why."""
    merged: list[Chunk] = []
    for c in chunks:
        if merged and merged[-1].token_count < min_tokens:
            prev = merged.pop()
            c = _combine(prev, c, doc_id)
        merged.append(c)
    if len(merged) > 1 and merged[-1].token_count < min_tokens:
        last = merged.pop()
        prev = merged.pop()
        merged.append(_combine(prev, last, doc_id))
    for i, c in enumerate(merged, start=1):
        c.chunk_id = f"{doc_id}-{i:04d}"
    return merged


def _combine(first: Chunk, second: Chunk, doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=first.chunk_id,
        doc_id=doc_id,
        text=f"{first.text}\n{second.text}",
        heading_trail=first.heading_trail,
        section_no=first.section_no,
        page_start=first.page_start,
        page_end=second.page_end,
        token_count=first.token_count + second.token_count,
    )


_HISTOGRAM_BUCKETS: list[tuple[str, int]] = [
    ("0", 0),
    ("1-25", 25),
    ("26-75", 75),
    ("76-150", 150),
    ("151-300", 300),
    ("301-500", 500),
    ("500+", float("inf")),
]


def token_histogram(chunks: list[Chunk]) -> dict[str, int]:
    """Bucket chunk token counts so the real distribution can be inspected
    before ever turning merging on."""
    buckets = {label: 0 for label, _ in _HISTOGRAM_BUCKETS}
    for c in chunks:
        for label, upper_bound in _HISTOGRAM_BUCKETS:
            if c.token_count <= upper_bound:
                buckets[label] += 1
                break
    return buckets


def load_body_pages(path: Path) -> list[tuple[int, str]]:
    """Convenience loader: page number (1-indexed) + text for every page
    after the cover sheet (page 1). For NHS/UK-government employer policy
    PDFs, which always have one."""
    with pdfplumber.open(path) as pdf:
        return [
            (i, page.extract_text() or "")
            for i, page in enumerate(pdf.pages, start=1)
            if i > 1
        ]


def load_all_pages(path: Path) -> list[tuple[int, str]]:
    """No cover-sheet skip — for GOV.UK statutory PDFs, which have none."""
    with pdfplumber.open(path) as pdf:
        return [
            (i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)
        ]
