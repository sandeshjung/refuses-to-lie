from pathlib import Path

from refuses_to_lie.intake import Supersession, parse_cover_sheet

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "employer"


def test_a36_employment_break_policy():
    cs = parse_cover_sheet(CORPUS / "a36-employment-break-policy-april-26.pdf")

    assert cs.doc_ref is None
    assert cs.title == "Employment Break Policy"
    assert cs.version == "5"
    assert cs.approval_date == "April 2023"
    assert cs.review_date == "January 2026"
    assert cs.expiry_date == "April 2026"
    assert cs.issued_by == "Kathryn Large, Senior HR Business Partner"
    assert cs.author == "Michelle O’Sullivan, Senior HR Advisor"
    assert cs.supersedes == [Supersession(name="Version 4", status=None, date=None)]
    assert cs.fields_not_found == ["doc_ref"]


def test_uhn_annual_leave_policy():
    cs = parse_cover_sheet(
        CORPUS / "uhn-annual-leave-for-agenda-for-change-staff000100pdf.pdf"
    )

    assert cs.doc_ref == "UHN-PO-HR24"
    assert (
        cs.title
        == "Annual Leave & Public Holiday Entitlements for Agenda for Change Staff Policy"
    )
    assert cs.version == "1"
    assert cs.approval_date == "March 2024"
    assert cs.review_date == "December 2026"
    assert cs.expiry_date == "March 2027"
    assert cs.issued_by == "Paula Kirkpatrick, Chief People Officer"
    assert cs.author == "Sarah Kinsella, HR Business Partner – Governance and Compliance"
    assert cs.supersedes == [
        Supersession(
            name="NGH – Annual Leave and General Public Holiday Entitlements NGH-PO-562",
            status=None,
            date=None,
        ),
        Supersession(
            name="KGH - Annual Leave for Non-Medical Staff Policy A.3.1",
            status=None,
            date=None,
        ),
    ]
    assert cs.fields_not_found == []


def test_b1_1_superseded_maintaining_high_professional_standards():
    cs = parse_cover_sheet(
        CORPUS / "b1.1-maintaining-high-professional-standards-oct-22-SUPERSEDED.pdf"
    )

    assert cs.doc_ref == "B1.1"
    assert cs.title == (
        "B.1.1. MAINTAINING HIGH PROFESSIONAL STANDARDS IN THE MODERN NHS: "
        "Conduct, Capability and Ill-Health, Policies and Procedures for "
        "Medical & Dental Practitioners"
    )
    assert cs.version == "2"
    assert cs.approval_date is None
    assert cs.review_date == "September 2021"
    assert cs.expiry_date == (
        "October 2022, extension agreed by OMG in light of new group policies"
    )
    assert cs.issued_by == "Mark Smith, Director of HR and OD"
    assert cs.author == "Nic Nicolaou"
    assert cs.fields_not_found == ["approval_date"]

    # The case most likely to silently break: a three-column Supersedes row
    # (policy name | status word | date) must parse into a single
    # Supersession with a non-None status and date, not a plain string.
    assert cs.supersedes == [
        Supersession(
            name="Maintaining High Professional Standards Policy",
            status="Expired",
            date="March 2018",
        )
    ]
    assert cs.supersedes[0].status == "Expired"
    assert cs.supersedes[0].date == "March 2018"


def test_uhn_flexible_retirement_policy_blank_doc_ref():
    # "Document Reference Number" is left blank in the source PDF; the raw
    # text stream runs straight into the next label ("Policy/Guideline").
    # Regression test for a bug where that next label's text was captured
    # as if it were the doc ref value.
    cs = parse_cover_sheet(CORPUS / "uhn-flexible-retirement-policy-mos-20112023pdf.pdf")

    assert cs.doc_ref is None
    assert "doc_ref" in cs.fields_not_found


def test_b11_maintaining_high_professional_standards_current():
    cs = parse_cover_sheet(CORPUS / "b11-maintaining-high-professional-standards-may-2027.pdf")

    assert cs.doc_ref == "UHN-PO-HR26"
    assert cs.title == (
        "MAINTAINING HIGH PROFESSIONAL STANDARDS IN THE MODERN NHS: Conduct, "
        "Capability and Ill-Health, Policies and Procedures for Medical & "
        "Dental Practitioners"
    )
    assert cs.version == "1"
    assert cs.approval_date == "May 2024"
    assert cs.review_date == "February 2027"
    assert cs.expiry_date == "May 2027"
    assert cs.issued_by is None
    assert cs.author == "Nic Nicolaou, Medical Workforce Manager"
    assert cs.supersedes == [
        Supersession(
            name=(
                "Maintaining High Professional Standards in the Modern NHS "
                "– Policy B.1.1 – KGH"
            ),
            status=None,
            date=None,
        )
    ]
    assert cs.fields_not_found == ["issued_by"]
