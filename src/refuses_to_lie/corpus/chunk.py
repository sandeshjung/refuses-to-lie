"""The citable unit this whole package produces."""

from __future__ import annotations

from dataclasses import dataclass


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
