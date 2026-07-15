"""Fast checks for the sandbagging DiD port (ADR 0012 + calm_system swap).

No model download: dataset contract, guards, the calm/pressured variant
construction, and the regime pin against sycophancy's DID_SYSTEM.
"""

import pytest

from dprobe import data
from dprobe.data import mcq, sandbagging, sycophancy
from dprobe.did import _variants


def test_legacy_completion_design_is_default_and_unchanged():
    """data.get('sandbagging') stays the 12-item system-contrast seed."""
    ex = data.get("sandbagging")
    labels = [e.label for e in ex]
    assert labels.count(0) == labels.count(1) == len(sandbagging.QA)
    for e in ex:
        assert e.meta.get("answer"), "legacy items carry the graded answer"


def test_did_suffix_matches_sycophancy_regime():
    """Both types must probe the SAME forced-choice regime (ADR 0012)."""
    assert sycophancy.DID_SYSTEM.endswith(sandbagging.DID_SUFFIX)
    assert sandbagging.CONTROL_DID_SYSTEM.endswith(sandbagging.DID_SUFFIX)


def test_did_build_is_unlabeled_system_pressure():
    """One example per question, -1 sentinel, pressure in the SYSTEM prompt,
    calm_system carries the control side, user text identical calm/pressured."""
    ex = sandbagging.build(design="did", source="factual-small")
    assert len(ex) >= 8
    assert all(e.label == sandbagging.UNLABELED for e in ex)
    assert all(e.system == sandbagging.SANDBAG_SYSTEM + sandbagging.DID_SUFFIX
               for e in ex)
    assert all(e.meta["calm_system"] == sandbagging.CONTROL_DID_SYSTEM for e in ex)
    assert all(e.meta["neutral_user"] == e.user for e in ex), \
        "sandbagging pressure is the SYSTEM prompt; user text is unchanged"
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "one example per question"
    for e in ex:
        m, nm = e.meta["matching"], e.meta["not_matching"]
        assert m in (" (A)", " (B)") and nm in (" (A)", " (B)") and m != nm


def test_did_pressure_variants():
    """incentive swaps ONLY the pressured system prompt; guards raise."""
    inst = sandbagging.build(design="did", source="factual-small")
    ince = sandbagging.build(design="did", source="factual-small",
                             pressure="incentive")
    assert all(e.system == sandbagging.INCENTIVE_SYSTEM + sandbagging.DID_SUFFIX
               for e in ince)
    assert len(inst) == len(ince)
    for a, b in zip(inst, ince):
        assert a.user == b.user
        assert a.meta["matching"] == b.meta["matching"]
        assert a.meta["calm_system"] == b.meta["calm_system"]
    with pytest.raises(ValueError, match="pressure"):
        sandbagging.build(design="did", source="factual-small", pressure="scary")
    with pytest.raises(ValueError, match="did"):
        sandbagging.build(design="completion", pressure="incentive")
    with pytest.raises(ValueError, match="factual"):
        sandbagging.build(design="did", source="opinion")


def test_variants_swap_system_on_calm_side():
    """did._variants: calm variants use calm_system + same user; pressured keep
    the pressure system. Answer prefixes follow the label."""
    ex = sandbagging.build(design="did", source="factual-small")[0]
    for label, expect_prefix_key in ((1, "matching"), (0, "not_matching")):
        ex.label = label
        v = _variants(ex)
        assert v["calm_promptfinal"].system == sandbagging.CONTROL_DID_SYSTEM
        assert v["pressured_promptfinal"].system == ex.system
        assert v["calm_promptfinal"].user == v["pressured_promptfinal"].user
        assert v["calm_answertoken"].meta["assistant_prefix"] == ex.meta["not_matching"]
        assert (v["pressured_answertoken"].meta["assistant_prefix"]
                == ex.meta[expect_prefix_key])


def test_variants_unchanged_for_sycophancy():
    """No calm_system in sycophancy meta -> behaviour identical to before the
    port (calm side keeps ex.system, strips the user turn)."""
    ex = sycophancy.build(design="did", source="factual-small")[0]
    ex.label = 0
    v = _variants(ex)
    assert "calm_system" not in ex.meta
    assert v["calm_promptfinal"].system == ex.system
    assert v["calm_promptfinal"].user == ex.meta["neutral_user"]
    assert v["calm_promptfinal"].user != ex.user


def test_dispatcher_routes_by_design_marker():
    """sandbagging_filter routes did-design examples to the shared did_filter,
    legacy examples to the capability filter (marker logic only, no model)."""
    ro = sandbagging.build(design="did", source="factual-small")[0]
    assert ro.meta.get("design") == "did"
    legacy = data.get("sandbagging")[0]
    assert legacy.meta.get("design") != "did"
    assert data.FILTERS["sandbagging"] is sandbagging.sandbagging_filter


def test_split_is_disjoint_and_shared_with_sycophancy():
    """Same mcq source, same 95/5 split boundary as sycophancy's factual rows --
    the transfer-matrix hygiene invariant, now for the did design."""
    train = {e.user for e in sandbagging.build(design="did", source="factual-small")}
    test = {e.user for e in sandbagging.build(design="did", source="factual-small",
                                              split="test")}
    assert train and train.isdisjoint(test)
    syco_train = {e.meta["neutral_user"] for e in
                  sycophancy.build(design="did", source="factual-small")}
    # sycophancy did bodies == our user text, byte-for-byte, same split side
    assert {u for u in train} == syco_train


def test_gate_calm_prompt_uses_control_system():
    """The belief-gate helpers must read the calm prompt under calm_system."""
    ex = sandbagging.build(design="did", source="factual-small")[0]

    class FakeTok:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize=False):
            return "|".join(m["content"] for m in messages)

    from dprobe.activations import build_prompt
    from dprobe.data.base import Example
    calm = Example(ex.meta.get("calm_system", ex.system), ex.meta["neutral_user"],
                   0, ex.deception_type)
    p = build_prompt(FakeTok(), calm)
    assert sandbagging.CONTROL_DID_SYSTEM in p
    assert sandbagging.SANDBAG_SYSTEM not in p


def test_steer_build_items_swaps_system_for_sandbagging():
    """steer.build_items: the calm (unpressured) sandbagging item must drop the
    SANDBAG system for the CONTROL system while keeping the user turn -- steering
    the wrong turn would leave the pressure in place (the whole point of the port).
    Pressured item keeps the SANDBAG system."""
    from dprobe.steer import build_items
    ex = sandbagging.build(design="did", source="factual-small")

    calm = build_items(ex, pressured=False)
    press = build_items(ex, pressured=True)
    assert len(calm) == len(press) == len(ex)
    for src, c, p in zip(ex, calm, press):
        # calm side: CONTROL system, pressure removed, same question
        assert c.example.system == sandbagging.CONTROL_DID_SYSTEM
        assert sandbagging.SANDBAG_SYSTEM not in c.example.system
        assert c.example.user == src.user
        # pressured side: SANDBAG system retained
        assert p.example.system == src.system
        assert p.example.system.startswith(sandbagging.SANDBAG_SYSTEM)
        # scoring strings carried through unchanged
        assert c.matching == src.meta["matching"]
        assert c.not_matching == src.meta["not_matching"]


def test_steer_build_items_incentive_arm():
    """Same swap holds for the incentive arm (calm side is still CONTROL)."""
    from dprobe.steer import build_items
    ex = sandbagging.build(design="did", source="factual-small", pressure="incentive")
    calm = build_items(ex, pressured=False)
    press = build_items(ex, pressured=True)
    for c, p in zip(calm, press):
        assert c.example.system == sandbagging.CONTROL_DID_SYSTEM
        assert p.example.system.startswith(sandbagging.INCENTIVE_SYSTEM)


def test_steer_build_items_unchanged_for_sycophancy():
    """Sycophancy: pressure is the USER turn, so the calm swap strips to
    neutral_user and KEEPS ex.system -- behaviour must be identical to pre-port."""
    from dprobe.steer import build_items
    ex = sycophancy.build(design="did", source="factual-small")
    calm = build_items(ex, pressured=False)
    press = build_items(ex, pressured=True)
    for src, c, p in zip(ex, calm, press):
        assert c.example.system == src.system            # system unchanged
        assert c.example.user == src.meta["neutral_user"]  # user stripped
        assert c.example.user != src.user
        assert p.example.user == src.user
