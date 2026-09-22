"""Stage 3: build the section tree and orchestrate a document's chunking.

A stack of currently-open sections, where pushing a heading at level L
closes any open sections at level >= L first (so a chunk's heading_trail
is always its true root-to-leaf ancestry).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from refuses_to_lie.corpus.chunk import Chunk
from refuses_to_lie.corpus.cleanup import is_toc_page, strip_boilerplate
from refuses_to_lie.corpus.headings import (
    document_uses_dotted_numbering,
    match_numbered_heading,
    match_unnumbered_heading,
    section_level,
)
from refuses_to_lie.corpus.postprocess import merge_short_chunks


def count_tokens(text: str) -> int:
    """Whitespace word count — a cheap proxy, not a real LLM tokenizer."""
    return len(text.split())


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


class SectionTree:
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

        numbered = match_numbered_heading(line, self._allow_plain_english)
        if numbered:
            section_no, heading = numbered
            self._open(section_level(section_no), heading, section_no, page_num)
            return

        if self._stack:
            unnumbered_heading = match_unnumbered_heading(line)
            if unnumbered_heading:
                nearest_numbered_level = self._nearest_numbered_level()
                self._open(nearest_numbered_level + 1, unnumbered_heading, None, page_num)
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

    def _open(self, level: int, heading: str, section_no: str | None, page_num: int) -> None:
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
        """A heading immediately followed by another heading with no body
        text between them (e.g. "10.1 PART I" then, on the very next line,
        "ACTION WHEN A CONCERN ARISES") should be folded into one heading —
        the ToC renders these as a single combined entry ("10.1 Part i:
        action when a concern arises").

        This only applies when the second heading is UNNUMBERED. A heading
        that carries its own number (e.g. "4.1 Annual Leave Entitlement"
        following an empty "4. SUBSTANTIVE CONTENT" container) is always a
        genuine, separately citable subsection and must open its own chunk
        even though its parent had no body of its own.
        """
        return (
            section_no is None
            and bool(self._stack)
            and self._stack[-1].is_empty
            and self._stack[-1].level < level
        )

    def _fold_into_top(self, heading: str, page_num: int) -> None:
        top = self._stack[-1]
        top.heading_trail = top.heading_trail[:-1] + (f"{top.heading_trail[-1]}: {heading}",)
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
                token_count=count_tokens(text),
            )
        )


def chunk_document(
    doc_id: str,
    pages: list[tuple[int, str]],
    min_tokens: int = 0,
) -> list[Chunk]:
    """Split a document's body pages into section-aware chunks.

    `pages` is a list of (page_num, page_text) for the BODY only (caller
    excludes the cover sheet, e.g. by skipping page 1). Splits on numbered
    headings ("1. Introduction", "10.1 PART I", "4.2. Annual Leave
    Entitlement") and on unnumbered bold/capitalised sub-headings, which are
    flattened to a single extra level beneath their nearest numbered
    ancestor (plain-text extraction can't reliably tell how deeply an
    unnumbered heading nests, so we don't fabricate depth we can't verify).

    By default (min_tokens=0) short sections are NOT merged into neighbors —
    a short clause with no concrete numbers should stay its own citable unit
    rather than being padded by an adjacent clause that has them, which would
    make near-miss hallucination questions look easier than they are. Pass
    min_tokens > 0 to opt into merging.
    """
    pages = strip_boilerplate(pages)
    allow_plain_english = not document_uses_dotted_numbering(pages)

    tree = SectionTree(doc_id, allow_plain_english=allow_plain_english)
    for page_num, text in pages:
        if is_toc_page(text):
            continue
        for raw_line in text.split("\n"):
            tree.add_line(raw_line.strip(), page_num)

    chunks = tree.finish()
    if min_tokens > 0:
        chunks = merge_short_chunks(chunks, min_tokens, doc_id)
    return chunks
