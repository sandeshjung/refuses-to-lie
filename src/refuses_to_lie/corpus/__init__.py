"""Turn a policy PDF's body text into section-aware, citable chunks.

This reads only the body (page 2 onward — the cover sheet is handled by
intake.py). It targets the NHS/UK-government "employer" policy corpus, whose
headings follow a numbered-outline style ("1. Introduction", "10.1 PART I",
"4.2. Annual Leave Entitlement") with occasional unnumbered bold/capitalised
sub-headings. The GOV.UK statutory corpus uses plain-English numbered
headings ("1. Overview", "2. What you'll get") instead of Title Case, which
is handled as a fallback gated on the document never using dotted (X.Y)
sub-numbering — see corpus.headings.

Package layout:
    chunk.py       - the Chunk dataclass this package produces
    cleanup.py     - stage 1: strip boilerplate / ToC pages
    headings.py    - stage 2: classify a line as heading or body text
    tree.py        - stage 3: build the section tree, chunk_document()
    postprocess.py - optional merging, token_histogram()
    loaders.py     - PDF -> page text
"""

from __future__ import annotations

from refuses_to_lie.corpus.chunk import Chunk
from refuses_to_lie.corpus.loaders import load_all_pages, load_body_pages
from refuses_to_lie.corpus.postprocess import token_histogram
from refuses_to_lie.corpus.tree import chunk_document

__all__ = [
    "Chunk",
    "chunk_document",
    "token_histogram",
    "load_body_pages",
    "load_all_pages",
]
