from pathlib import Path

from refuses_to_lie.pipeline import load_employer_chunks, load_statutory_chunks

EMPLOYER_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"
STATUTORY_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "statutory"


def test_employer_doc_id_uses_cover_sheet_doc_ref_when_present():
    chunks = load_employer_chunks(EMPLOYER_CORPUS)
    doc_ids = {c.doc_id for c in chunks}

    # uhn-annual-leave's cover sheet has a real doc_ref (see test_intake.py).
    assert "UHN-PO-HR24" in doc_ids


def test_employer_doc_id_falls_back_to_filename_when_doc_ref_missing():
    chunks = load_employer_chunks(EMPLOYER_CORPUS)
    doc_ids = {c.doc_id for c in chunks}

    # a36 has no "Document Reference Number" field at all (see test_intake.py).
    assert "a36-employment-break-policy-april-26" in doc_ids


def test_statutory_doc_id_is_filename_stem():
    chunks = load_statutory_chunks(STATUTORY_CORPUS)
    doc_ids = {c.doc_id for c in chunks}

    assert "Print Statutory Sick Pay (SSP) - GOV.UK" in doc_ids
