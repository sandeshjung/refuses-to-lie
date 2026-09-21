from __future__ import annotations

from pathlib import Path

from refuses_to_lie.corpus import Chunk, chunk_document, load_all_pages, load_body_pages
from refuses_to_lie.intake import parse_cover_sheet


def _employer_doc_id(path: Path) -> str:
    cover_sheet = parse_cover_sheet(path)
    return cover_sheet.doc_ref or path.stem


def load_employer_chunks(employer_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(employer_dir.glob("*.pdf")):
        doc_id = _employer_doc_id(pdf_path)
        chunks.extend(chunk_document(doc_id, load_body_pages(pdf_path)))
    return chunks


def load_statutory_chunks(statutory_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(statutory_dir.glob("*.pdf")):
        chunks.extend(chunk_document(pdf_path.stem, load_all_pages(pdf_path)))
    return chunks


def load_corpus_chunks(employer_dir: Path, statutory_dir: Path) -> list[Chunk]:
    return load_employer_chunks(employer_dir) + load_statutory_chunks(statutory_dir)
