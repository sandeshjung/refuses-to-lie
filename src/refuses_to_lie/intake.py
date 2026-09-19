from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

_REQUIRED_FIELDS = (
    "doc_ref",
    "title",
    "version",
    "approval_date",
    "review_date",
    "expiry_date",
    "issued_by",
    "author",
)

_STATUS_WORDS = {"expired", "superseded", "archived", "withdrawn", "retired"}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)
_DATE_RE = re.compile(
    rf"^\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{4}}$"
    rf"|^(?:{_MONTHS})\s+\d{{4}}$"
    rf"|^\d{{4}}$",
    re.IGNORECASE,
)
_NEW_ENTRY_PREFIX_RE = re.compile(r"^[A-Z]{2,6}\s*[-–—]")
_HEADER_FRAGMENT_RE = re.compile(r"^[A-Z][a-z]+$")
_DOC_REF_NUMBER_RE = re.compile(
    r"Document Reference Number\s*[:\-–—]?\s*([^\n]+)"
)

_KNOWN_LABEL_TEXT = {
    "policy/guideline",
    "policy/guideline title",
    "policy title",
    "executive summary",
    "supersedes",
    "description of amendment(s)",
    "this policy will impact on",
    "financial implications",
    "policy area",
    "version number",
    "issued by",
    "expiry date",
    "approval date",
    "review date",
    "author",
    "impact assessment",
    "impact assessment date",
    "document reference",
}


def _looks_like_label(value_norm_lower: str) -> bool:
    if _classify_label(value_norm_lower) is not None:
        return True
    return any(
        value_norm_lower == label or label.startswith(value_norm_lower)
        for label in _KNOWN_LABEL_TEXT
    )


@dataclass
class Supersession:
    name: str
    status: str | None = None
    date: str | None = None


@dataclass
class CoverSheet:
    doc_ref: str | None = None
    title: str | None = None
    version: str | None = None
    approval_date: str | None = None
    review_date: str | None = None
    expiry_date: str | None = None
    issued_by: str | None = None
    author: str | None = None
    supersedes: list[Supersession] = field(default_factory=list)
    fields_not_found: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _classify_label(norm_lower: str) -> str | None:
    if "title" in norm_lower and "policy" in norm_lower:
        return "title"
    if norm_lower.startswith("document reference"):
        return "doc_ref"
    if "supersede" in norm_lower:
        return "supersedes"
    if norm_lower == "version number":
        return "version"
    if norm_lower == "approval date":
        return "approval_date"
    if norm_lower == "review date":
        return "review_date"
    if norm_lower == "issued by":
        return "issued_by"
    if norm_lower == "expiry date":
        return "expiry_date"
    if norm_lower == "author":
        return "author"
    return None


def _is_boundary(normalized: str) -> bool:
    if normalized.endswith(":"):
        return True
    if normalized.lower() in _STATUS_WORDS:
        return False
    return bool(_HEADER_FRAGMENT_RE.fullmatch(normalized))


def _is_duplicate(normalized_lower: str, last_accepted: str | None) -> bool:
    if last_accepted is None:
        return False
    if normalized_lower == last_accepted:
        return True
    return last_accepted.startswith(normalized_lower) or normalized_lower.startswith(
        last_accepted
    )


def _parse_supersedes(segments: list[str]) -> list[Supersession]:
    lines: list[str] = []
    for segment in segments:
        lines.extend(line.strip() for line in segment.split("\n") if line.strip())

    entries: list[Supersession] = []
    current: Supersession | None = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if lower in _STATUS_WORDS:
            if current is not None:
                current.status = line
            continue
        if _DATE_RE.match(line):
            if current is not None:
                current.date = line
            continue
        no_lowercase = not any(c.islower() for c in line)
        starts_new = i == 0 or (
            not no_lowercase and _NEW_ENTRY_PREFIX_RE.match(line)
        )
        if starts_new or current is None:
            current = Supersession(name=line)
            entries.append(current)
        else:
            current.name = f"{current.name} {line}"
    return entries


def _extract_standalone_doc_ref(page_text: str) -> str | None:
    match = _DOC_REF_NUMBER_RE.search(page_text)
    if not match:
        return None
    value = match.group(1).strip()
    if not value:
        return None
    if _looks_like_label(_normalize(value).lower()):
        return None
    return value


def parse_cover_sheet(path: Path) -> CoverSheet:
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        page_text = page.extract_text() or ""
        tables = page.extract_tables()

    sheet = CoverSheet()
    field_values: dict[str, str] = {}
    supersedes_segments: list[str] = []

    if tables:
        table = tables[0]
        current_field: str | None = None
        last_accepted: str | None = None

        for row in table:
            for cell in row:
                if not cell or not cell.strip():
                    continue
                normalized = _normalize(cell)
                normalized_lower = normalized.lower()

                if _is_duplicate(normalized_lower, last_accepted):
                    continue

                if _is_boundary(normalized):
                    current_field = _classify_label(normalized_lower.rstrip(":"))
                    last_accepted = normalized_lower
                    continue

                last_accepted = normalized_lower

                if current_field is None:
                    continue
                if current_field == "supersedes":
                    supersedes_segments.append(cell)
                    continue

                if current_field in field_values:
                    field_values[current_field] += " " + normalized
                else:
                    field_values[current_field] = normalized

    sheet.doc_ref = field_values.get("doc_ref")
    sheet.title = field_values.get("title")
    sheet.version = field_values.get("version")
    sheet.approval_date = field_values.get("approval_date")
    sheet.review_date = field_values.get("review_date")
    sheet.expiry_date = field_values.get("expiry_date")
    sheet.issued_by = field_values.get("issued_by")
    sheet.author = field_values.get("author")
    sheet.supersedes = _parse_supersedes(supersedes_segments)

    if not sheet.doc_ref:
        sheet.doc_ref = _extract_standalone_doc_ref(page_text)

    sheet.fields_not_found = [
        name for name in _REQUIRED_FIELDS if not getattr(sheet, name)
    ]
    if not sheet.supersedes:
        sheet.fields_not_found.append("supersedes")

    return sheet
