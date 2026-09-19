from pathlib import Path

from refuses_to_lie.corpus import chunk_document, load_body_pages, token_histogram

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


def _chunks(doc_id: str, filename: str):
    return chunk_document(doc_id, load_body_pages(CORPUS / filename))


def test_a36_employment_break_policy_section_1():
    chunks = _chunks("A.3.6", "a36-employment-break-policy-april-26.pdf")
    intro = next(c for c in chunks if c.section_no == "1")

    assert intro.heading_trail == ("Introduction",)
    assert intro.page_start == 3
    assert intro.cite_label == "A.3.6 §1 (p. 3)"
    assert intro.text.startswith(
        "This policy details the employment break scheme for employees of"
    )
    assert intro.token_count > 0


def test_uhn_annual_leave_two_level_heading_trail():
    chunks = _chunks(
        "UHN-PO-HR24", "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"
    )
    entitlement = next(c for c in chunks if c.section_no == "4.1")

    # Root-to-leaf: "4. SUBSTANTIVE CONTENT" itself has no body of its own
    # (it goes straight into its first subsection), so only the subsection
    # is emitted as a chunk — but its trail still carries the full ancestry.
    assert entitlement.heading_trail == (
        "SUBSTANTIVE CONTENT",
        "Annual Leave Entitlement",
    )
    assert entitlement.cite_label == "UHN-PO-HR24 §4.1 (p. 5)"
    assert entitlement.text.startswith(
        "The annual leave year runs from 1st April to 31st March"
    )
    assert entitlement.token_count > 0


def test_b1_1_and_b11_same_content_different_section_numbers():
    # b1.1 and b11 are the same policy lineage (old vs new version) with the
    # same "Part I: Action When a Concern Arises" content, but b1.1 numbers
    # it "10.1" (under top-level "10.0 Policy Framework") while b11 renumbers
    # it "4.1" (under top-level "4. Policy in Practice"). A regex that
    # hardcodes either leading digit would silently break one of these.
    b1_1_chunks = _chunks(
        "B1.1",
        "b1.1-maintaining-high-professional-standards-oct-22-SUPERSEDED.pdf",
    )
    b11_chunks = _chunks(
        "UHN-PO-HR26", "b11-maintaining-high-professional-standards-may-2027.pdf"
    )

    b1_1_part_i = next(
        c for c in b1_1_chunks if c.heading_trail[-1].startswith("PART I:")
    )
    b11_part_i = next(
        c for c in b11_chunks if c.heading_trail[-1].startswith("PART I:")
    )

    assert b1_1_part_i.section_no == "10.1"
    assert b11_part_i.section_no == "4.1"
    assert b1_1_part_i.section_no != b11_part_i.section_no

    assert b1_1_part_i.heading_trail == (
        "POLICY FRAMEWORK",
        "PART I: ACTION WHEN A CONCERN ARISES: INTRODUCTION",
    )
    assert b11_part_i.heading_trail == (
        "POLICY IN PRACTICE",
        "PART I: ACTION WHEN A CONCERN ARISES: Introduction",
    )

    assert b1_1_part_i.cite_label == "B1.1 §10.1 (p. 6)"
    assert b11_part_i.cite_label == "UHN-PO-HR26 §4.1 (p. 5)"

    for chunk in (b1_1_part_i, b11_part_i):
        assert chunk.text.startswith(
            "1. The management of performance is a continuous process"
        )
        assert chunk.token_count > 0


def test_full_employer_corpus_token_histogram():
    all_chunks = []
    for pdf_path in sorted(CORPUS.glob("*.pdf")):
        pages = load_body_pages(pdf_path)
        all_chunks.extend(chunk_document(pdf_path.stem, pages))

    assert all_chunks, "expected at least one chunk across the employer corpus"
    for chunk in all_chunks:
        assert chunk.text.strip(), f"empty chunk text: {chunk.chunk_id}"

    histogram = token_histogram(all_chunks)
    print(f"\ntotal chunks: {len(all_chunks)}")
    print(f"token histogram: {histogram}")

    assert sum(histogram.values()) == len(all_chunks)
