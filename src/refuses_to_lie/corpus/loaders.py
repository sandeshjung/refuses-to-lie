"""PDF page loaders."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


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
        return [(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]
