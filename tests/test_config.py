from refuses_to_lie.config import RunConfig

def test_defaults():
    cfg = RunConfig(id="A", label="baseline")
    assert cfg.retrieval == "dense"
    assert cfg.rerank is False

def test_frozen():
    cfg = RunConfig(id="A", label="baseline")
    try:
        cfg.rerank = True
        assert False, "should not be able to mutate a frozen dataclass"
    except AttributeError:
        pass

def test_ladder_changes_one_axis_at_a_time():
    from dataclasses import fields
    from refuses_to_lie.config import LADDER

    axis_fields = {"retrieval", "rerank", "require_citations", "verifier", "abstain"}
    constant_fields = {
        f.name for f in fields(LADDER[0])
        if f.name not in axis_fields and f.name not in ("id", "label")
    }

    for lo, hi in zip(LADDER, LADDER[1:]):
        changed_axis = {f for f in axis_fields if getattr(lo, f) != getattr(hi, f)}
        assert len(changed_axis) == 1, (
            f"{lo.id} -> {hi.id} changes {changed_axis}, expected exactly one axis field"
        )
        for f in constant_fields:
            assert getattr(lo, f) == getattr(hi, f), (
                f"{lo.id} -> {hi.id} moved held-constant field {f!r}"
            )