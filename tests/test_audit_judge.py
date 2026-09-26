import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "audit_judge", Path(__file__).resolve().parent.parent / "scripts" / "audit_judge.py"
)
audit_judge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_judge)


def test_kappa_is_one_for_perfect_agreement():
    pairs = [("CORRECT", "CORRECT"), ("PARTIAL", "PARTIAL"), ("INCORRECT", "INCORRECT")]
    assert audit_judge._kappa(pairs) == 1.0


def test_kappa_discounts_agreement_that_label_frequency_predicts():
    # A judge that says CORRECT to everything agrees often on a set that is
    # mostly correct -- and kappa must not reward that.
    pairs = [("CORRECT", "CORRECT")] * 8 + [("INCORRECT", "CORRECT")] * 2
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    assert raw == 0.8
    assert audit_judge._kappa(pairs) <= 0.0
