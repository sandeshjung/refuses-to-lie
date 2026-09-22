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
INJECTED = CORPUS / "injected"

INJECTION_STYLES = {"direct", "hidden", "metadata"}


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
    # The whole point of this category is that stale content is also
    # retrievable and disagrees with the current answer — that has to be
    # tracked as structured data (which document is the trap), not left only
    # in free-text notes, or an eval harness can't check it automatically.
    #
    # A question whose stale content sits INSIDE the correct document (an
    # expired commitment a current policy still describes as in force) has no
    # separate distractor. That case must be declared with an explicit flag
    # rather than by omitting the field, so silence stays an error.
    real_ids = _all_real_doc_ids()
    for q in _load():
        if q["category"] != "contradictory_stale":
            continue
        if q.get("distractor_is_same_document"):
            assert "distractor_doc_ids" not in q, (
                f"{q['id']}: same-document staleness can't also name a separate distractor"
            )
            continue
        distractors = q.get("distractor_doc_ids")
        assert distractors, f"{q['id']} has no distractor_doc_ids"
        for doc_id in distractors:
            assert doc_id in real_ids, f"{q['id']} references unknown doc_id {doc_id!r}"
        assert set(distractors).isdisjoint(q["expected_doc_ids"]), (
            f"{q['id']}: a distractor can't also be the correct grounding doc"
        )


def _injected_doc_ids() -> set[str]:
    return {p.stem for p in INJECTED.glob("*.pdf")}


def test_prompt_injection_questions_name_their_attack_document():
    # An injection question is only meaningful if we know which document
    # carries the payload, what style of attack it is, and what following it
    # would look like in the output. Without the last of those, "did the
    # injection work?" is a judgement call rather than something the harness
    # can score.
    injected = _injected_doc_ids()
    for q in _load():
        if q["category"] != "prompt_injection":
            continue
        attack_docs = q.get("injection_doc_ids")
        assert attack_docs, f"{q['id']} has no injection_doc_ids"
        for doc_id in attack_docs:
            assert doc_id in injected, f"{q['id']} references unknown injected doc {doc_id!r}"
        assert q.get("injection_style") in INJECTION_STYLES, (
            f"{q['id']} has unknown injection_style {q.get('injection_style')!r}"
        )
        assert q.get("injection_behaviour"), (
            f"{q['id']} does not say what compliance looks like"
        )
        assert set(attack_docs).isdisjoint(q["expected_doc_ids"]), (
            f"{q['id']}: the attack document can't also be the correct grounding doc"
        )


def test_prompt_injection_answerability_is_self_consistent():
    for q in _load():
        if q["category"] != "prompt_injection":
            continue
        if q["expected_answerable"]:
            assert q["expected_doc_ids"], f"{q['id']} is answerable but has no grounding"
            assert q["expected_answer_summary"], f"{q['id']} is answerable but has no summary"
        else:
            assert q["expected_doc_ids"] == [], (
                f"{q['id']} is unanswerable but names grounding"
            )
            assert q["expected_answer_summary"] is None


def test_every_injected_document_is_exercised_by_a_question():
    # A planted attack document nothing ever asks about measures nothing.
    used = {d for q in _load() for d in q.get("injection_doc_ids", [])}
    unused = _injected_doc_ids() - used
    assert not unused, f"injected documents no question targets: {sorted(unused)}"


def test_all_three_injection_styles_are_represented():
    styles = {q["injection_style"] for q in _load() if q["category"] == "prompt_injection"}
    assert styles == INJECTION_STYLES, f"missing injection styles: {INJECTION_STYLES - styles}"


def test_duplicate_question_text_is_only_an_intentional_clean_attacked_pair():
    # Two questions share wording on purpose: a near_miss and the
    # prompt_injection question that asks the same thing with an adversarial
    # document planted, so the pair measures what the injection costs. Any
    # OTHER duplicate is an accident, and would quietly double-weight one fact.
    seen: dict[str, list[dict]] = {}
    for q in _load():
        seen.setdefault(q["question"].strip().lower(), []).append(q)

    for text, group in seen.items():
        if len(group) == 1:
            continue
        assert len(group) == 2, f"question asked {len(group)} times: {text!r}"
        categories = {q["category"] for q in group}
        assert categories == {"near_miss", "prompt_injection"}, (
            f"unexpected duplicate {text!r} across categories {categories}"
        )
        ids = {q["id"] for q in group}
        for q in group:
            partner = q.get("paired_with")
            assert partner and partner in ids - {q["id"]}, (
                f"{q['id']} shares wording with its pair but does not declare paired_with"
            )


def test_every_question_has_reviewable_notes():
    # The notes field is how a human verifies 240 questions by skimming, so an
    # empty or one-line note defeats the point of storing them.
    for q in _load():
        assert len(q["notes"]) >= 60, f"{q['id']} has notes too thin to review"
