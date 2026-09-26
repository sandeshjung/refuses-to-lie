"""Which documents are allowed to answer questions at all.

The injection experiment showed the attack does not work by getting the
model to follow instructions -- it never did, across 150 attempts. It works
by source capture: an adversarial page sits in the corpus, retrieves well
because it was written to be topically dense, and the model reports its
fabricated figures as policy. Nothing in rungs A-F asks whether a retrieved
document deserves to be believed.

So the defence lives upstream of the model: a register of the documents
that went through document control, with a content hash for each. A chunk
from an unregistered document never reaches the context window.

Be clear about what this does and does not demonstrate. In this experiment
the injected documents were never registered, so the register excludes
them by construction and the headline number is not evidence of anything
on its own. What it does establish is (a) the fix is architectural, not a
prompt, and (b) what filtering costs on the clean corpus. The residual risk
moves to the register's integrity -- an attacker who can get a document
registered, or alter a registered file, is not stopped here, which is why
`find_tampering` checks hashes rather than trusting file names.

The weaker alternative -- trusting any document that LOOKS controlled,
i.e. carries a document-control cover sheet -- is what `looks_controlled`
implements, so that its forgeability can be measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from refuses_to_lie.intake import parse_cover_sheet
from refuses_to_lie.pipeline import employer_doc_id, statutory_doc_id


@dataclass(frozen=True)
class RegisteredDocument:
    doc_id: str
    source: str
    file: str
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_register(employer_dir: Path, statutory_dir: Path) -> dict[str, RegisteredDocument]:
    """Register every document in the curated corpus, keyed by indexed doc_id.

    Raises on a doc_id collision: two files sharing an id would mean one of
    them is registered under the other's identity, which is precisely the
    ambiguity a register exists to remove.
    """
    register: dict[str, RegisteredDocument] = {}
    sources = (
        ("employer", employer_dir, employer_doc_id),
        ("statutory", statutory_dir, statutory_doc_id),
    )
    for source, directory, derive_id in sources:
        for pdf_path in sorted(directory.glob("*.pdf")):
            doc_id = derive_id(pdf_path)
            if doc_id in register:
                claimed_by = register[doc_id].file
                raise ValueError(
                    f"doc_id {doc_id!r} claimed by both {claimed_by} and {pdf_path.name}"
                )
            register[doc_id] = RegisteredDocument(
                doc_id, source, pdf_path.name, _sha256(pdf_path)
            )
    return register


def save_register(register: dict[str, RegisteredDocument], path: Path) -> None:
    payload = {doc_id: asdict(doc) for doc_id, doc in sorted(register.items())}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_register(path: Path) -> dict[str, RegisteredDocument]:
    return {
        doc_id: RegisteredDocument(**doc)
        for doc_id, doc in json.loads(path.read_text()).items()
    }


def trusted_doc_ids(path: Path) -> frozenset[str]:
    return frozenset(load_register(path))


def find_tampering(register_path: Path, employer_dir: Path, statutory_dir: Path) -> list[str]:
    """Registered documents whose file is missing or no longer matches its hash.

    A register that only checks names is defeated by editing a registered
    PDF in place; the hash is what makes the register mean anything.
    """
    directories = {"employer": employer_dir, "statutory": statutory_dir}
    problems = []
    for doc in load_register(register_path).values():
        path = directories[doc.source] / doc.file
        if not path.exists():
            problems.append(f"{doc.doc_id}: {doc.file} is missing")
        elif _sha256(path) != doc.sha256:
            problems.append(f"{doc.doc_id}: {doc.file} has changed since it was registered")
    return problems


def looks_controlled(pdf_path: Path) -> bool:
    """The naive trust rule: the document carries a document-control cover sheet.

    Requires both a document reference and an approval date, the two fields
    a real controlled policy always states. Exists to be measured, not used:
    an attacker who writes a convincing cover sheet passes it, and the
    metadata-dressed injection documents were built to do exactly that.
    """
    cover = parse_cover_sheet(pdf_path)
    return bool(cover.doc_ref and cover.approval_date)
