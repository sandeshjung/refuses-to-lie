import shutil
from pathlib import Path

import pytest

from refuses_to_lie.pipeline import load_corpus_chunks
from refuses_to_lie.provenance import (
    build_register,
    find_tampering,
    save_register,
    trusted_doc_ids,
)

ROOT = Path(__file__).resolve().parent.parent
EMPLOYER = ROOT / "corpus" / "employer"
STATUTORY = ROOT / "corpus" / "statutory"
REGISTER = ROOT / "corpus" / "register.json"


def test_committed_register_covers_every_indexed_document():
    # If a real document were missing, rung H would silently delete it from
    # every answer and the clean-corpus cost of provenance would be wrong.
    indexed = {c.doc_id for c in load_corpus_chunks(EMPLOYER, STATUTORY)}
    assert indexed <= trusted_doc_ids(REGISTER)


def test_committed_register_matches_the_files_on_disk():
    assert find_tampering(REGISTER, EMPLOYER, STATUTORY) == []


def test_no_injected_document_is_registered():
    injected = {p.stem for p in (ROOT / "corpus" / "injected").glob("*.pdf")}
    assert not injected & trusted_doc_ids(REGISTER)


def _mini_corpus(tmp_path: Path) -> tuple[Path, Path]:
    employer, statutory = tmp_path / "employer", tmp_path / "statutory"
    employer.mkdir()
    statutory.mkdir()
    source = sorted(STATUTORY.glob("*.pdf"))[0]
    shutil.copy(source, statutory / source.name)
    return employer, statutory


def test_editing_a_registered_file_is_detected(tmp_path: Path):
    # A register that checked only names would be defeated by rewriting a
    # registered PDF in place; the hash is what makes it mean anything.
    employer, statutory = _mini_corpus(tmp_path)
    register_path = tmp_path / "register.json"
    save_register(build_register(employer, statutory), register_path)
    assert find_tampering(register_path, employer, statutory) == []

    target = next(statutory.glob("*.pdf"))
    target.write_bytes(target.read_bytes() + b"\n% appended")
    assert (
        "changed since it was registered"
        in find_tampering(register_path, employer, statutory)[0]
    )


def test_a_deleted_registered_file_is_detected(tmp_path: Path):
    employer, statutory = _mini_corpus(tmp_path)
    register_path = tmp_path / "register.json"
    save_register(build_register(employer, statutory), register_path)

    next(statutory.glob("*.pdf")).unlink()
    assert "missing" in find_tampering(register_path, employer, statutory)[0]


def test_two_files_claiming_one_id_is_refused(tmp_path: Path):
    employer, statutory = _mini_corpus(tmp_path)
    source = next(statutory.glob("*.pdf"))
    # Same stem in both directories -> same doc_id from two different files.
    employer_copy = employer / source.name
    shutil.copy(source, employer_copy)
    with pytest.raises(ValueError, match="claimed by both"):
        build_register(employer, statutory)
