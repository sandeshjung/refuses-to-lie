import json
from pathlib import Path

EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.json"

VALID_CATEGORIES = {
    "answerable_single",
    "answerable_synthesis",
    "near_miss",
    "out_of_scope",
    "false_premise",
    "contradictory_stale",
    "prompt_injection",
}

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_answerable",
    "expected_doc_ids",
    "expected_answer_summary",
    "notes",
}

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _load() -> list[dict]:
    return json.loads(EVAL_FILE.read_text())


def _all_real_doc_ids() -> set[str]:
    # Cheap structural check: every doc_id an eval question claims to be
    # grounded in must correspond to an actual PDF filename stem (or, for
    # employer docs, the doc_ref used in intake tests) — not a typo.
    stems = set()
    for pdf in (CORPUS / "employer").glob("*.pdf"):
        stems.add(pdf.stem)
    for pdf in (CORPUS / "statutory").glob("*.pdf"):
        stems.add(pdf.stem)
    # Known doc_ref overrides used by pipeline.py for employer docs whose
    # cover sheet has a real Document Reference Number (see test_pipeline.py).
    stems |= {
        "UHN-PO-HR24",
        "UHN-PO-HR26",
        "UHN-PO-HR07",
        "UHN-PO-HR10",
        "UHN-PO-HR18",
        "UHN-PO-HR87",
        "UHN-PO-122",
        "B1.1",
    }
    return stems


def test_eval_file_is_valid_json_and_non_empty():
    questions = _load()
    assert questions


def test_every_question_has_required_fields():
    for q in _load():
        assert set(q.keys()) >= REQUIRED_FIELDS, f"{q.get('id')} missing fields"


def test_every_question_id_is_unique():
    ids = [q["id"] for q in _load()]
    assert len(ids) == len(set(ids))


def test_every_category_is_known():
    for q in _load():
        assert q["category"] in VALID_CATEGORIES, f"{q['id']} has unknown category"


def test_unanswerable_questions_have_no_expected_doc_ids_or_summary():
    # near_miss, out_of_scope, and false_premise questions must not claim a
    # grounded answer exists (false_premise may still reference the doc the
    # false premise is ABOUT, without claiming an answer summary).
    for q in _load():
        if q["category"] in ("near_miss", "out_of_scope"):
            assert q["expected_doc_ids"] == []
            assert q["expected_answer_summary"] is None
        if q["category"] in ("near_miss", "out_of_scope", "false_premise"):
            assert q["expected_answerable"] is False


def test_answerable_questions_have_grounding_and_summary():
    for q in _load():
        if q["category"] in (
            "answerable_single",
            "answerable_synthesis",
            "contradictory_stale",
        ):
            assert q["expected_answerable"] is True
            assert q["expected_doc_ids"], f"{q['id']} has no grounding doc_ids"
            assert q["expected_answer_summary"], f"{q['id']} has no answer summary"


def test_synthesis_questions_cite_at_least_two_documents():
    for q in _load():
        if q["category"] == "answerable_synthesis":
            assert len(q["expected_doc_ids"]) >= 2, f"{q['id']} isn't cross-document"


def test_expected_doc_ids_correspond_to_real_documents():
    real_ids = _all_real_doc_ids()
    for q in _load():
        for doc_id in q["expected_doc_ids"]:
            assert doc_id in real_ids, f"{q['id']} references unknown doc_id {doc_id!r}"


def test_contradictory_stale_questions_name_a_distractor_document():
    # The whole point of this category is that a stale/conflicting document
    # is also retrievable and disagrees with the current one — that has to
    # be tracked as structured data (which doc is the trap), not left only
    # in free-text notes, or an eval harness can't check it automatically.
    real_ids = _all_real_doc_ids()
    for q in _load():
        if q["category"] == "contradictory_stale":
            distractors = q.get("distractor_doc_ids")
            assert distractors, f"{q['id']} has no distractor_doc_ids"
            for doc_id in distractors:
                assert doc_id in real_ids, f"{q['id']} references unknown doc_id {doc_id!r}"
            assert set(distractors).isdisjoint(q["expected_doc_ids"]), (
                f"{q['id']}: a distractor can't also be the correct grounding doc"
            )
