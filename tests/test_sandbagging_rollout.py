"""Fast checks for the sandbagging rollout design (ADR 0011). No model download.

Mirrors tests/test_smoke.py: dataset contract, guards, and the cross-type
invariants that make the shared MCQ source safe for the transfer matrix.
"""

import pytest

from dprobe import data
from dprobe.data import mcq, rollout, sandbagging, sycophancy


def test_legacy_completion_design_is_default_and_unchanged():
    """data.get('sandbagging') stays the 12-item system-contrast seed (back-compat)."""
    ex = data.get("sandbagging")
    labels = [e.label for e in ex]
    assert labels.count(0) == labels.count(1) == len(sandbagging.QA)
    for e in ex:
        assert e.meta.get("answer"), "legacy items carry the graded answer"
        assert e.system in (sandbagging.CONTROL_SYSTEM, sandbagging.SANDBAG_SYSTEM)


def test_rollout_design_requires_mcq_source():
    """Guard: the rollout design needs the MCQ sources (ADR 0011)."""
    with pytest.raises(ValueError, match="factual"):
        sandbagging.build(design="rollout", source="opinion")
    with pytest.raises(ValueError, match="rollout"):
        sandbagging.build(design="framing")


def test_rollout_build_is_unlabeled_single_prompt_pressure():
    """Rollout build (offline fixture): one example per question, -1 sentinel,
    no completion, SANDBAG system on every example -- both future classes share
    the identical prompt, so there is nothing lexical to shortcut on."""
    ex = sandbagging.build(design="rollout", source="factual-small")
    assert len(ex) >= 8
    assert all(e.label == rollout.UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.system == sandbagging.SANDBAG_SYSTEM for e in ex)
    assert all(e.deception_type == "sandbagging" for e in ex)
    assert all(e.meta.get("design") == "rollout" for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "one example per question"
    for e in ex:
        m, nm = e.meta["matching"], e.meta["not_matching"]
        assert m in (" (A)", " (B)") and nm in (" (A)", " (B)") and m != nm
        # matching = the WRONG answer (label 1 = sandbagged); check against body
        assert e.meta["neutral_user"] == e.user, \
            "sandbagging pressure is the SYSTEM prompt; user text is unchanged"
        assert e.user.endswith("\n\nAnswer:")


def test_gate_prompt_swaps_system_not_user():
    """The capability gate reads the same question under the CONTROL system."""
    ex = sandbagging.build(design="rollout", source="factual-small")[0]
    neutral = sandbagging._neutral_example(ex)
    assert neutral.system == sandbagging.CONTROL_SYSTEM
    assert neutral.user == ex.user
    assert neutral.system != ex.system


def test_mcq_split_is_disjoint_and_stable():
    train = list(mcq.fixture_rows("train"))
    test = list(mcq.fixture_rows("test"))
    train_q = {r["question"] for r in train}
    test_q = {r["question"] for r in test}
    assert train_q and train_q.isdisjoint(test_q)
    # deterministic across calls
    assert [r["idx"] for r in mcq.fixture_rows("train")] == [r["idx"] for r in train]


def test_mcq_letter_side_is_balanced_and_deterministic():
    tags = {" (A)": 0, " (B)": 0}
    for i in range(200):
        body, correct_tag, wrong_tag = mcq.two_choice_body("Q?", "yes", "no", i)
        assert body.endswith("\n\nAnswer:") and correct_tag != wrong_tag
        assert f"{correct_tag} yes" in body and f"{wrong_tag} no" in body
        assert (body, correct_tag, wrong_tag) == mcq.two_choice_body("Q?", "yes", "no", i)
        tags[correct_tag] += 1
    assert 60 <= tags[" (A)"] <= 140, f"letter side should be ~50/50, got {tags}"


def test_cross_type_bodies_and_split_agree_with_sycophancy():
    """THE transfer-matrix hygiene invariant: for the same source row, the MCQ
    body equals sycophancy's factual `neutral_user` byte-for-byte, and a
    question is in the same train/test split for both types."""
    sand = {e.meta["neutral_user"] for e in
            sandbagging.build(design="rollout", source="factual-small", split="train")}
    syco = {e.meta["neutral_user"] for e in
            sycophancy.build(design="rollout", source="factual-small", split="train")}
    assert sand == syco, "same bodies, same split side, byte-for-byte"

    sand_test = {e.meta["neutral_user"] for e in
                 sandbagging.build(design="rollout", source="factual-small", split="test")}
    assert sand.isdisjoint(sand_test)


def test_shared_engine_reuses_sycophancy_parse_choice():
    """One parser implementation: the engine imports sycophancy.parse_choice."""
    assert rollout.parse_choice is sycophancy.parse_choice


def test_engine_defaults_match_sycophancy_knobs():
    """The shared engine starts from the same knob values as sycophancy's module,
    so a sandbagging run and a sycophancy run differ only where ADR 0011 says."""
    for knob in ("ROLLOUT_N", "ROLLOUT_TEMPERATURE", "ROLLOUT_MAX_NEW_TOKENS",
                 "ROLLOUT_PREFIX_MODE", "GATE_MODE", "GATE_N", "GATE_THRESHOLD",
                 "GATE_TEMPERATURE"):
        assert getattr(rollout, knob) == getattr(sycophancy, knob), knob


def test_engine_render_log_shows_set_membership():
    """Per-run log: gate failures + per-question outcomes (pure function)."""
    from dprobe.data.base import Example

    stats = {"questions_in": 3, "ambivalent_questions": 2, "n_examples": 2}
    gate_failed = [Example("s", "Q3 body", -1, "sandbagging",
                           meta={"neutral_user": "What is 2+2?"})]
    q_records = [
        {"question": "Is 17 prime?", "user": "Q1 body", "deceptive_letter": "A",
         "parsed": ["deceptive", "honest", "honest"], "kept_pairs": 1,
         "outcome": "ambivalent"},
        {"question": "Is 2 even?", "user": "Q2 body", "deceptive_letter": "B",
         "parsed": ["honest", "honest", "honest"], "kept_pairs": 0,
         "outcome": "single-class"},
    ]
    log = rollout.render_log(stats, gate_failed, q_records, {"Q1 body"},
                             "model cannot answer correctly even unpressured")
    assert "gate-fail" in log and "What is 2+2?" in log
    assert "cannot answer correctly" in log
    assert "USED" in log and "Is 17 prime?" in log
    assert "SINGLE-CLASS" in log and "Is 2 even?" in log
    assert "questions_in: 3" in log


def test_dispatcher_routes_by_design_marker():
    """sandbagging_filter must dispatch rollout examples to the engine (checked
    here only via the marker logic -- the engine itself needs a model)."""
    from dprobe.data.base import Example

    legacy = Example(sandbagging.CONTROL_SYSTEM, "q", 0, "sandbagging",
                     meta={"answer": "4"})
    assert legacy.meta.get("design") != "rollout"
    ro = sandbagging.build(design="rollout", source="factual-small")[0]
    assert ro.meta.get("design") == "rollout"
    assert data.FILTERS["sandbagging"] is sandbagging.sandbagging_filter


def test_verify_read_positions_accepts_sandbagging_prefix():
    """The read-position guard works unchanged for sandbagging rollout children
    (the ' (X)' commit prefix mechanism is type-agnostic)."""
    from dprobe.activations import verify_read_positions
    from dprobe.data.base import Example

    class FakeTok:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize=False):
            return " ".join(m["content"] for m in messages) + " ASSISTANT:"

        def __call__(self, text, return_tensors=None):
            words = text.split()
            class _R:
                input_ids = [words]
            return _R()

        def decode(self, ids):
            return " ".join(ids) if isinstance(ids, list) else str(ids)

    ok = Example(sandbagging.SANDBAG_SYSTEM, "Q body", 1, "sandbagging",
                 meta={"assistant_prefix": " (B)"})
    verify_read_positions(FakeTok(), [ok], verbose=False)
