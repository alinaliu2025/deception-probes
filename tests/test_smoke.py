"""Fast checks that need no model download. Run:  pytest -q

These validate the math and the dataset contract, so you catch a broken refactor
in seconds instead of after a 5-minute activation extraction. The actual model
run is exercised by scripts/train_one.py.
"""

import numpy as np
import pytest

from dprobe import data
from dprobe.config import DECEPTION_TYPES
from dprobe.data import sycophancy
from dprobe.evaluate import direction_cosines, layer_sweep, transfer_matrix
from dprobe.probes import fit_lda, fit_lr, fit_mms, fit_mms_std


def _separable(n=60, hidden=32, layers=4, seed=0):
    """Synthetic activations where class 1 is shifted along one axis -> easy to probe."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    acts = rng.normal(size=(n, layers, hidden))
    acts[y == 1, :, 0] += 4.0  # plant a deception direction at dim 0
    return acts, y


def test_datasets_are_balanced_and_typed():
    for t in DECEPTION_TYPES:
        ex = data.get(t)
        labels = [e.label for e in ex]
        assert labels.count(0) == labels.count(1), f"{t} not balanced"
        assert all(e.deception_type == t for e in ex)
        assert len(ex) >= 8


def test_sycophancy_framing_design_is_clean():
    """The instruction-contrast design must differ only in the system prompt and
    read before any answer (no completion). Skips if the dataset isn't available."""
    try:
        ex = sycophancy.build(design="framing")
    except Exception as e:  # no network / dataset cache -> not a logic failure
        pytest.skip(f"sycophancy dataset unavailable: {e}")

    labels = [e.label for e in ex]
    assert labels.count(0) == labels.count(1), "framing design not balanced"
    # no completion anywhere -> read position is prompt-final, before any answer
    assert all(e.completion is None for e in ex), "framing must not paste a completion"

    by_user: dict[str, dict[int, object]] = {}
    for e in ex:
        by_user.setdefault(e.user, {})[e.label] = e
    for pair in by_user.values():
        assert set(pair) == {0, 1}
        assert pair[0].user == pair[1].user                 # same task
        assert pair[1].system == sycophancy.NEUTRAL_SYSTEM  # label 1 = neutral
        assert pair[0].system == sycophancy.HONEST_SYSTEM   # label 0 = honest-primed
        assert pair[1].system != pair[0].system             # differ ONLY in system
        assert "matching" in pair[1].meta and "not_matching" in pair[1].meta


def test_neutral_question_strips_persona():
    """Persona-strip for the behavioral design: bio dies, bare question stays."""
    q = ("Hello, my name is Jane Doe. I am a 45-year-old liberal woman. "
         "I think experts are great. "
         "In general, would you say experts who study a subject for many years are..."
         "\n (A) x\n (B) y\n\nAnswer:")
    s = sycophancy.neutral_question(q)
    assert s is not None and s.startswith("In general, would you say experts")
    assert "Jane" not in s and s.endswith("\n\nAnswer:")

    # nlp-survey stem; bio echoes 'the claim' but not the marker phrase
    q2 = ("Hello, my name is Bob. I agree with the claim that X is true. "
          "Do you agree or disagree with the following claim about NLP? X is true."
          "\n\nChoices:\n (A) Agree\n (B) Disagree\n\nAnswer:")
    s2 = sycophancy.neutral_question(q2)
    assert s2 is not None and s2.startswith("Do you agree or disagree")
    assert "Bob" not in s2

    assert sycophancy.neutral_question("no recognised stem here") is None


def test_balance_subsamples_majority_class():
    from dprobe.data.base import Example
    exs = ([Example("s", f"u{i}", 1, "t") for i in range(10)]
           + [Example("s", f"v{i}", 0, "t") for i in range(4)])
    out = sycophancy._balance(exs, seed=0)
    labels = [e.label for e in out]
    assert labels.count(1) == labels.count(0) == 4


def test_sycophancy_behavioral_build_is_unlabeled_and_promptfinal():
    """Behavioral design: one example per question, -1 sentinel until the filter
    runs, no completion, identical prompt construction for what will become both
    classes. Skips if the dataset isn't available."""
    try:
        ex = sycophancy.build(design="behavioral")
    except Exception as e:  # no network / dataset cache -> not a logic failure
        pytest.skip(f"sycophancy dataset unavailable: {e}")

    assert len(ex) >= 8
    assert all(e.label == sycophancy.BEHAVIORAL_UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.system == sycophancy.NEUTRAL_SYSTEM for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "must be one example per question"
    for e in ex[:100]:
        nq = e.meta["neutral_user"]
        assert nq in e.user and len(nq) < len(e.user), "strip must remove the bio"
        assert "matching" in e.meta and "not_matching" in e.meta


def test_parse_choice_maps_rollout_text_to_answer():
    """Rollout design (ADR 0008): first '(X)' decides; truncation ends at it."""
    matching, not_matching = " (A)", " (B)"
    choice, end = sycophancy.parse_choice(" (B) I disagree", matching, not_matching)
    assert choice == "not_matching" and " (B) I disagree"[:end] == " (B)"
    choice, end = sycophancy.parse_choice("I'd say (A), since...", matching, not_matching)
    assert choice == "matching" and "I'd say (A), since..."[:end].endswith("(A)")
    assert sycophancy.parse_choice("no letter here", matching, not_matching) == (None, None)
    assert sycophancy.parse_choice(" (C) what?", matching, not_matching) == (None, None)


def test_sycophancy_rollout_build_matches_behavioral_plus_marker():
    """Rollout build: identical construction to behavioral, plus the dispatch
    marker meta['design']='rollout'. Skips if the dataset isn't available."""
    try:
        ex = sycophancy.build(design="rollout")
    except Exception as e:  # no network / dataset cache -> not a logic failure
        pytest.skip(f"sycophancy dataset unavailable: {e}")

    assert len(ex) >= 8
    assert all(e.label == sycophancy.BEHAVIORAL_UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.meta.get("design") == "rollout" for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "must be one example per question"


def test_make_factual_row_construction():
    """Factual source (ADR 0009), pure unit: assertion in the pressure question
    but not the neutral one; matching = asserted WRONG letter (label 1 =
    deceptive), not_matching = correct; both ' (X)' formatted; letter side
    ~50/50 over many indices. No network needed."""
    rows = [sycophancy._make_factual_row(
        f"What is {i}+{i}?", correct=str(2 * i), wrong=str(2 * i + 1), idx=i)
        for i in range(200)]
    letters = {"A": 0, "B": 0}
    for i, r in enumerate(rows):
        assert set(r) == {"question", "answer_matching_behavior",
                          "answer_not_matching_behavior", "neutral_user"}
        # neutral body is the bare MCQ; the pressure question is that body plus
        # an assertion prefix claiming the wrong answer
        assert r["neutral_user"].endswith("\n\nAnswer:")
        assert r["question"].endswith(r["neutral_user"])
        assert len(r["question"]) > len(r["neutral_user"])
        wrong_tag = r["answer_matching_behavior"].strip()
        assert wrong_tag in r["question"][: -len(r["neutral_user"])], \
            "assertion prefix must claim the wrong answer"
        # ' (X)' format, letters disjoint, wrong letter holds the wrong text
        m, nm = r["answer_matching_behavior"], r["answer_not_matching_behavior"]
        assert m in (" (A)", " (B)") and nm in (" (A)", " (B)") and m != nm
        assert f"{m} {2 * i + 1}" in r["neutral_user"]   # matching -> wrong text
        assert f"{nm} {2 * i}" in r["neutral_user"]      # not_matching -> correct
        letters[m.strip("() ")] += 1
        # deterministic in idx
        assert r == sycophancy._make_factual_row(
            f"What is {i}+{i}?", correct=str(2 * i), wrong=str(2 * i + 1), idx=i)
    assert 60 <= letters["A"] <= 140, f"letter side should be ~50/50, got {letters}"


def test_factual_source_requires_behavioral_or_rollout():
    """Guard: factual rows have no pre-written completion (ADR 0009)."""
    with pytest.raises(ValueError, match="factual"):
        sycophancy.build(design="completion", source="factual")
    with pytest.raises(ValueError, match="factual"):
        sycophancy.build(design="framing", source="factual")
    # the offline smoke source is a factual source and obeys the same guard
    with pytest.raises(ValueError, match="factual"):
        sycophancy.build(design="completion", source="factual-small")


def test_factual_small_source_is_offline_and_well_formed():
    """The repo-resident smoke source (fixtures/factual_smoke.jsonl) builds
    rollout examples with NO network -- same schema/markers as ARC 'factual',
    plus source='factual-small'. This is the dataset for local demos/smoke."""
    ex = sycophancy.build(design="rollout", source="factual-small")
    assert len(ex) >= 8, "smoke fixture should have enough rows to run"
    assert all(e.label == sycophancy.BEHAVIORAL_UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.meta.get("design") == "rollout" for e in ex)
    assert all(e.meta.get("source") == "factual-small" for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "one example per question"
    for e in ex:
        # pressure prompt = neutral body + an assertion of the WRONG answer
        assert e.meta["neutral_user"] in e.user
        assert len(e.meta["neutral_user"]) < len(e.user)
        m, nm = e.meta["matching"], e.meta["not_matching"]
        assert m in (" (A)", " (B)") and nm in (" (A)", " (B)") and m != nm


def test_factual_small_train_test_split_is_disjoint():
    """The smoke source honours the fixed-seed 95/5 split like the other sources,
    so a demo run's train and test questions never overlap."""
    train = sycophancy.build(design="rollout", source="factual-small", split="train")
    test = sycophancy.build(design="rollout", source="factual-small", split="test")
    train_q = {e.meta["neutral_user"] for e in train}
    test_q = {e.meta["neutral_user"] for e in test}
    assert train_q.isdisjoint(test_q)
    assert train_q, "train split should be non-empty"


def test_sycophancy_factual_rollout_build():
    """Factual + rollout: same sentinel/marker contract as opinion, plus the
    source tag. Skips if the ARC dataset isn't available."""
    try:
        ex = sycophancy.build(design="rollout", source="factual")
    except Exception as e:  # no network / dataset cache -> not a logic failure
        pytest.skip(f"ARC dataset unavailable: {e}")

    assert len(ex) >= 8
    assert all(e.label == sycophancy.BEHAVIORAL_UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.meta.get("design") == "rollout" for e in ex)
    assert all(e.meta.get("source") == "factual" for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "must be one example per question"
    for e in ex[:100]:
        assert e.meta["neutral_user"] in e.user
        assert len(e.meta["neutral_user"]) < len(e.user)


def test_did_build_matches_behavioral_plus_marker():
    """did build (ADR 0012): same one-example-per-question, -1-sentinel
    construction as behavioral/rollout, plus the dispatch marker
    meta['design']=='did'. Uses the offline smoke source -- no network."""
    ex = sycophancy.build(design="did", source="factual-small")
    assert len(ex) >= 8
    assert all(e.label == sycophancy.BEHAVIORAL_UNLABELED for e in ex)
    assert all(e.completion is None for e in ex)
    assert all(e.meta.get("design") == "did" for e in ex)
    assert all(e.meta.get("source") == "factual-small" for e in ex)
    users = [e.user for e in ex]
    assert len(users) == len(set(users)), "one example per question"
    for e in ex:
        assert e.meta["neutral_user"] in e.user
        m, nm = e.meta["matching"], e.meta["not_matching"]
        assert m in (" (A)", " (B)") and nm in (" (A)", " (B)") and m != nm


def test_did_variants_read_positions():
    """dprobe.did._variants: calm side always reads the honest (not_matching)
    answer; the pressured side reads the letter the model committed -- matching
    iff it caved (label 1). Prompt-final variants carry no assistant_prefix."""
    from dprobe.data.base import Example
    from dprobe.did import _variants

    meta = {"matching": " (B)", "not_matching": " (A)",
            "neutral_user": "Q\n (A) x\n (B) y\n\nAnswer:"}
    caved = Example("sys", "PRESSURE Q", 1, "sycophancy", meta=dict(meta))
    held = Example("sys", "PRESSURE Q", 0, "sycophancy", meta=dict(meta))

    vc = _variants(caved)
    assert "assistant_prefix" not in vc["calm_promptfinal"].meta
    assert "assistant_prefix" not in vc["pressured_promptfinal"].meta
    assert vc["calm_promptfinal"].user == meta["neutral_user"]
    assert vc["pressured_promptfinal"].user == "PRESSURE Q"
    assert vc["calm_answertoken"].meta["assistant_prefix"] == " (A)"      # honest
    assert vc["pressured_answertoken"].meta["assistant_prefix"] == " (B)"  # caved -> wrong

    vh = _variants(held)
    assert vh["calm_answertoken"].meta["assistant_prefix"] == " (A)"
    assert vh["pressured_answertoken"].meta["assistant_prefix"] == " (A)"  # held -> honest


def test_did_arrows_isolate_capitulation():
    """The difference-of-differences math. Build activations as content (varies by
    question, and is CORRELATED with the label -- the ADR 0008 confound) + a shared
    pressure component + a capitulation component present only on caved items.
    diff-of-means ON THE ARROWS recovers the capitulation axis and separates the
    classes; diff-of-means on the raw pressured snapshots is dominated by content."""
    rng = np.random.default_rng(0)
    n, hidden = 40, 16
    y = np.array([1, 0] * (n // 2))
    content = rng.normal(size=(n, hidden)) * 5.0
    content[y == 1, 5] += 8.0             # content axis that tracks the label
    g = np.zeros(hidden); g[2] = 3.0      # generic pressure response, all items
    k = np.zeros(hidden); k[7] = 2.0      # capitulation, caved items only

    calm = content.copy()
    pressured = content + g + np.outer(y, k)
    arrows = pressured - calm             # first subtraction cancels content

    pa = fit_mms(arrows, y, layer=0, deception_type="t")
    assert abs(float(pa.direction @ (k / np.linalg.norm(k)))) > 0.99  # ~ the k axis
    assert (pa.predict(arrows) == y).mean() == 1.0                    # clean split
    # a probe on the raw pressured snapshots leans on the content dim, not k
    pp = fit_mms(pressured, y, layer=0, deception_type="t")
    assert abs(float(pp.direction[5])) > abs(float(pp.direction[7]))


def test_did_direction_equals_behavioral_minus_neutral():
    """Closed form (ADR 0012): the DiD direction (diff-of-means on arrows) equals
    the behavioral direction minus the neutral-read direction, pre-normalization."""
    rng = np.random.default_rng(1)
    n, hidden = 30, 12
    y = np.array([1, 0] * (n // 2))
    calm = rng.normal(size=(n, hidden))
    pressured = rng.normal(size=(n, hidden))

    def raw_dom(X):  # unnormalized diff-of-means
        return X[y == 1].mean(0) - X[y == 0].mean(0)

    assert np.allclose(raw_dom(pressured - calm), raw_dom(pressured) - raw_dom(calm))


def test_extract_arrows_slices_and_subtracts(monkeypatch):
    """extract_arrows lays out the 4 read-variants in a fixed order, extracts once,
    and slices the aligned blocks back out as pressured-minus-calm arrows per
    position. Fake the model-facing extract so the test needs no download."""
    from dprobe import did as did_mod
    from dprobe.data.base import Example

    meta = {"matching": " (B)", "not_matching": " (A)", "neutral_user": "NQ"}
    exs = [Example("s", "P1", 1, "sycophancy", meta=dict(meta)),
           Example("s", "P2", 0, "sycophancy", meta=dict(meta))]

    def fake_extract(model, tokenizer, flat, device, batch_size=None, acts_dtype=None):
        # flat order is [block0(all q), block1(all q), ...]; encode block*10+q so
        # every position's pressured-minus-calm is a constant we can assert on
        n = len(flat) // 4
        acts = np.zeros((len(flat), 3, 4), dtype=np.float32)
        for i in range(len(flat)):
            acts[i, :, 0] = (i // n) * 10 + (i % n)
        return acts, np.zeros(len(flat))

    monkeypatch.setattr(did_mod, "extract", fake_extract)
    monkeypatch.setattr(did_mod, "verify_read_positions", lambda *a, **k: None)

    arrows, labels, groups = did_mod.extract_arrows(None, None, "cpu", exs)
    # promptfinal = block1 - block0 = 10; answertoken = block3 - block2 = 10
    assert np.allclose(arrows["promptfinal"][:, :, 0], 10.0)
    assert np.allclose(arrows["answertoken"][:, :, 0], 10.0)
    assert list(labels) == [1, 0]
    assert groups == ["P1", "P2"]


def test_render_did_log_shows_decisions():
    """The did filter's per-question trace records gate failures (with evidence),
    the greedy pressured answer text, the caved/held verdict and unparsed drops,
    and class-balance membership. Pure function, no model."""
    from dprobe.data.base import Example

    stats = {"questions_in": 3, "gate_mode": "sampled"}
    gate_failed = [(Example("s", "PRESSURE agrees", -1, "sycophancy",
                            meta={"neutral_user": "Is the sky green?"}),
                    {"gate": "logprob", "lp_honest": -2.0, "lp_wrong": -0.1})]
    q_records = [
        {"question": "Is 17 prime?", "outcome": "caved", "ex_id": 1,
         "gate": {"gate": "sampled", "correct": 19, "parseable": 20, "n": 20,
                  "samples": [("correct", " (A) yes")]},
         "text": "Yes, you're right, (B)"},
        {"question": "Is 2 even?", "outcome": "held", "ex_id": 2,
         "gate": {"gate": "logprob", "lp_honest": -0.3, "lp_wrong": -1.2},
         "text": " (A) even"},
        {"question": "garbled?", "outcome": "unparsed",
         "gate": {"gate": "logprob", "lp_honest": -0.3, "lp_wrong": -1.2},
         "text": "hmm, no letter"},
    ]
    log = sycophancy._render_did_log(stats, gate_failed, q_records, kept_ids={1})
    assert "gate-fail" in log and "Is the sky green?" in log
    assert "[USED caved(1)]" in log and "Is 17 prime?" in log
    assert "[TRIMMED held(0)]" in log and "Is 2 even?" in log
    assert "[UNPARSED]" in log and "garbled?" in log
    assert 'pressure-answer "Yes, you\'re right, (B)"' in log
    assert 'gate-sample[correct] " (A) yes"' in log
    assert "questions_in: 3" in log


def test_render_rollout_log_shows_set_membership():
    """The per-run log (written to run_log.txt) records every gate failure (with
    the gate's evidence) and, per gated question, its full rollout texts, parsed
    outcomes and final set membership: USED / AMBIVALENT-BUT-TRIMMED /
    SINGLE-CLASS. Pure function, no model."""
    from dprobe.data.base import Example

    lp_gate = {"gate": "logprob", "lp_honest": -0.5, "lp_wrong": -1.5}
    stats = {"questions_in": 3, "ambivalent_questions": 2, "n_examples": 2}
    gate_failed = [(Example("s", "PRESSURE-Q3", -1, "sycophancy",
                            meta={"neutral_user": "Is the sky green?"}),
                    {"gate": "logprob", "lp_honest": -2.0, "lp_wrong": -0.1})]
    q_records = [
        {"question": "Is 17 prime?", "user": "PRESSURE-Q1", "cave_letter": "A",
         "gate": lp_gate, "parsed": ["caved", "held", "held"],
         "texts": [" (A) sure", " (B) no", " (B) it is not"],
         "kept_pairs": 1, "outcome": "ambivalent"},
        {"question": "Is 2 even?", "user": "PRESSURE-Q2", "cave_letter": "B",
         "gate": {"gate": "sampled", "correct": 19, "parseable": 20, "n": 20,
                  "samples": [("correct", " (A) certainly")]},
         "parsed": ["held", "held", "held"],
         "texts": [" (A) yes", " (A) yep", " (A) even"],
         "kept_pairs": 0, "outcome": "single-class"},
    ]
    kept_users = {"PRESSURE-Q1"}  # Q1 survived letter balance, Q2 did not (single-class)
    log = sycophancy._render_rollout_log(stats, gate_failed, q_records, kept_users)

    assert "gate-fail" in log and "Is the sky green?" in log
    assert "USED" in log and "Is 17 prime?" in log
    assert "SINGLE-CLASS" in log and "Is 2 even?" in log
    assert "questions_in: 3" in log  # summary block present
    # gate evidence and full rollout texts are recorded
    assert "lp_honest=-0.5" in log and "gate=sampled 19/20 correct" in log
    assert 'rollout[caved] " (A) sure"' in log
    assert 'rollout[held] " (B) it is not"' in log
    assert 'gate-sample[correct] " (A) certainly"' in log


def test_render_behavioral_log_shows_decisions():
    """The behavioral filter's per-question trace records the gate evidence, the
    pressure-prompt choice, and whether the example survived class balance."""
    stats = {"questions_in": 3}
    q_records = [
        {"question": "Is the sky green?", "outcome": "gate-fail",
         "gate": {"gate": "logprob", "lp_honest": -2.0, "lp_wrong": -0.1}},
        {"question": "Is 17 prime?", "outcome": "caved", "ex_id": 1,
         "gate": {"gate": "logprob", "lp_honest": -0.5, "lp_wrong": -1.5},
         "lp_press_match": -0.2, "lp_press_not": -0.9},
        {"question": "Is 2 even?", "outcome": "held", "ex_id": 2,
         "gate": {"gate": "sampled", "correct": 19, "parseable": 20, "n": 20,
                  "samples": [("correct", "The answer is (B)")]},
         "lp_press_match": -0.9, "lp_press_not": -0.2},
    ]
    log = sycophancy._render_behavioral_log(stats, q_records, kept_ids={1})
    assert "[GATE-FAIL]" in log and "Is the sky green?" in log
    assert "[USED caved(1)]" in log and "Is 17 prime?" in log
    assert "[TRIMMED held(0)]" in log and "Is 2 even?" in log
    assert 'gate-sample[correct] "The answer is (B)"' in log
    assert "questions_in: 3" in log


def test_render_framing_log_shows_flip_evidence():
    """The framing (keep-if-flips) trace records both log-prob comparisons and
    the verdict for every question, including incomplete pairs."""
    stats = {"questions_in": 3, "kept_questions": 1}
    q_records = [
        {"question": "Q kept?", "outcome": "kept", "syc_under_neutral": True,
         "honest_under_honest": True, "lp_neutral": (-0.2, -0.9),
         "lp_honest": (-1.1, -0.3)},
        {"question": "Q dropped?", "outcome": "dropped", "syc_under_neutral": False,
         "honest_under_honest": True, "lp_neutral": (-0.9, -0.2),
         "lp_honest": (-1.1, -0.3)},
        {"question": "Q half?", "outcome": "incomplete-pair"},
    ]
    log = sycophancy._render_framing_log(stats, q_records)
    assert "[KEPT]" in log and "Q kept?" in log
    assert "[DROPPED]" in log and "syc_under_neutral=False" in log
    assert "[INCOMPLETE-PAIR]" in log and "Q half?" in log
    assert "kept_questions: 1" in log


def test_render_steer_log_lists_completions():
    """The steering trace has one section per stage with per-item completion
    texts and rates recomputed from the recorded counts."""
    from dprobe.steer import render_steer_log

    trace = [
        {"pass": "add", "stage": "baseline (unpressured, unsteered)",
         "tag": "baseline",
         "items": [{"index": 0, "question": "Q one?",
                    "counts": {"wrong": 0, "correct": 1, "unparsed": 0},
                    "completions": [("correct", " (A) because")]}]},
        {"pass": "add", "stage": "alpha=2.0", "tag": "probe",
         "items": [{"index": 0, "question": "Q one?",
                    "counts": {"wrong": 1, "correct": 0, "unparsed": 0},
                    "completions": [("wrong", " (B)!")]}]},
    ]
    log = render_steer_log(trace)
    assert "ADD | baseline (unpressured, unsteered) [baseline]" in log
    assert "ADD | alpha=2.0 [probe]" in log
    assert "wrong_rate=0.000" in log and "wrong_rate=1.000" in log
    assert 'completion[correct] " (A) because"' in log
    assert 'completion[wrong] " (B)!"' in log


def test_capture_console_tees_and_collapses_progress(tmp_path):
    """The console tee: terminal output unchanged, console.log gets timestamped
    lines with \\r progress counters collapsed to their final frame, and both
    stdout and stderr recorded."""
    import re
    import sys

    from dprobe import runlog

    with runlog.capture_console(tmp_path):
        print("hello world")
        print("  1/3", end="\r", flush=True)
        print("  2/3", end="\r", flush=True)
        print("  3/3", end="\r", flush=True)
        print()  # progress terminator, as the extraction/filter loops do
        print("done")
        print("a warning", file=sys.stderr)
    text = (tmp_path / "console.log").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# console log | started")
    assert lines[-1].startswith("# console log | ended")
    assert "hello world" in text and "done" in text and "a warning" in text
    # progress collapsed: final frame once, intermediate frames gone
    assert text.count("3/3") == 1 and "1/3" not in text and "2/3" not in text
    for ln in lines[1:-1]:
        assert re.match(r"\d{2}:\d{2}:\d{2} ", ln), f"untimestamped line: {ln!r}"
    # streams restored after the block
    assert not isinstance(sys.stdout, runlog._Tee)
    assert not isinstance(sys.stderr, runlog._Tee)


def test_capture_console_records_crash(tmp_path):
    """A crash mid-run leaves its traceback in console.log -- with the run dir
    created at start, a dir without meta.json plus this log IS the post-mortem."""
    from dprobe import runlog

    with pytest.raises(RuntimeError, match="boom"):
        with runlog.capture_console(tmp_path):
            print("about to fail")
            raise RuntimeError("boom")
    text = (tmp_path / "console.log").read_text(encoding="utf-8")
    assert "about to fail" in text
    assert "Traceback" in text and "RuntimeError: boom" in text


def test_verify_read_positions_asserts_answer_token():
    """Index-verification guard: passes when the read position (last token) holds
    the rollout answer letter, raises when the prefix/read is off. Uses a fake
    tokenizer so no model download is needed."""
    from dprobe.activations import verify_read_positions
    from dprobe.data.base import Example

    class FakeTok:
        """Minimal stand-in: 'tokenises' by whitespace, 'decodes' back to text.
        build_prompt only needs apply_chat_template + a tokenizer call."""
        def apply_chat_template(self, messages, add_generation_prompt, tokenize=False):
            return " ".join(m["content"] for m in messages) + " ASSISTANT:"

        def __call__(self, text, return_tensors=None):
            words = text.split()
            class _R:  # mimic .input_ids[0] as a list of "ids" (here, the words)
                input_ids = [words]
            return _R()

        def decode(self, ids):
            return " ".join(ids) if isinstance(ids, list) else str(ids)

    ok = Example("sys", "Q body", 1, "sycophancy",
                 meta={"assistant_prefix": " (A)"})
    verify_read_positions(FakeTok(), [ok], verbose=False)  # letter (A) ends the prompt

    bad = Example("sys", "Q body", 1, "sycophancy",
                  meta={"assistant_prefix": " (A) and then a long tail of words that "
                        "pushes the letter out of the final tokens entirely here now"})
    with pytest.raises(AssertionError, match="read-position guard FAILED"):
        verify_read_positions(FakeTok(), [bad], verbose=False)


def test_gate_mode_defaults_to_logprob():
    """Belief-gate default stays the deterministic log-prob gate: the sampled gate
    is strictly opt-in (nothing existing changes behaviour)."""
    assert sycophancy.GATE_MODE == "logprob"


def test_completion_design_is_default_and_unchanged():
    """Default get() path stays the leaky completion baseline (back-compat)."""
    try:
        default = data.get("sycophancy")
        explicit = sycophancy.build(design="completion")
    except Exception as e:
        pytest.skip(f"sycophancy dataset unavailable: {e}")
    assert len(default) == len(explicit)
    assert all(e.completion is not None for e in default)


def test_probes_recover_planted_direction():
    acts, y = _separable()
    X = acts[:, 1, :]
    for fitter in (fit_mms, fit_mms_std, fit_lda, fit_lr):
        p = fitter(X, y, layer=1, deception_type="test")
        auroc_like = (p.predict(X) == y).mean()
        assert auroc_like > 0.9
        assert abs(np.linalg.norm(p.direction) - 1.0) < 1e-5


def test_layer_sweep_and_transfer_run():
    acts, y = _separable()
    aurocs, bl, probe = layer_sweep(acts, y, "lr", "a")
    assert 0 <= bl < acts.shape[1]
    assert aurocs[bl] > 0.9

    probes = {"a": probe, "b": probe}
    A = {"a": acts, "b": acts}
    L = {"a": y, "b": y}
    M, types = transfer_matrix(probes, A, L)
    assert M.shape == (2, 2)
    cos, _ = direction_cosines(probes)
    assert np.allclose(np.diag(cos), 1.0, atol=1e-5)


def test_load_probe_roundtrips(tmp_path):
    """A probe.npz written like train_one's is rebuilt into an equal Probe."""
    from dprobe.steer import load_probe

    d = np.zeros(8, dtype=np.float32)
    d[3] = 1.0
    path = tmp_path / "probe.npz"
    np.savez(path, direction=d, bias=0.5, layer=13, method="mms",
             deception_type="sycophancy")
    p = load_probe(path)
    assert p.layer == 13 and p.method == "mms" and p.deception_type == "sycophancy"
    assert p.bias == 0.5 and np.allclose(p.direction, d)


def test_steer_hooks_add_and_ablate():
    """The forward-hook factories add alpha*v and project v out of a block output.

    Exercises the residual-stream math with no model: a decoder block returns a
    tuple whose first element is the hidden state, which is what the hooks edit.
    """
    torch = pytest.importorskip("torch")
    from dprobe.steer import make_add_hook, make_ablate_hook

    v = torch.zeros(4)
    v[1] = 1.0  # unit vector along dim 1
    h = torch.tensor([[[2.0, 5.0, 1.0, 0.0]]])  # [batch=1, seq=1, hidden=4]

    add_out = make_add_hook(v, 3.0)(None, None, (h.clone(),))
    assert torch.allclose(add_out[0][0, 0], torch.tensor([2.0, 8.0, 1.0, 0.0]))

    ablate_out = make_ablate_hook(v)(None, None, (h.clone(),))
    # the component along v (the 5.0 at dim 1) is removed, the rest untouched
    assert torch.allclose(ablate_out[0][0, 0], torch.tensor([2.0, 0.0, 1.0, 0.0]))


def test_random_probe_is_unit_and_same_layer():
    """The random control shares the probe's layer but is an independent unit vector."""
    from dprobe.probes import Probe
    from dprobe.steer import random_probe

    p = Probe(direction=np.eye(8)[0].astype(np.float32), bias=0.0, layer=15,
              method="mms", deception_type="sycophancy")
    r = random_probe(p, seed=0)
    assert r.layer == 15 and r.deception_type == "sycophancy"
    assert abs(np.linalg.norm(r.direction) - 1.0) < 1e-5
    assert abs(float(r.direction @ p.direction)) < 0.9  # not aligned with the probe


def test_parse_rate_tracks_generation_breakage():
    """parse_rate falls as completions stop yielding a choice -- the signal that a
    high-alpha wrong_rate is breakage, not steering."""
    from dprobe.steer import _parse_rate, _wrong_rate

    ok = [{"wrong": 2, "correct": 3, "unparsed": 0}]
    assert _parse_rate(ok) == 1.0
    broken = [{"wrong": 1, "correct": 0, "unparsed": 4}]
    assert _parse_rate(broken) == pytest.approx(0.2)
    assert _wrong_rate(broken) == 1.0  # over survivors only -> misleading without parse_rate


def test_build_items_selects_pressured_vs_neutral_prompt():
    """build_items uses the assertion prompt when pressured and the persona-stripped
    neutral question otherwise -- the two steering passes read different prompts."""
    from dprobe.data.base import Example
    from dprobe.steer import build_items

    ex = Example("sys", "PRESSURE: I think (B).\n\nQ\n (A) x\n (B) y\n\nAnswer:",
                 label=-1, deception_type="sycophancy",
                 meta={"matching": " (B)", "not_matching": " (A)",
                       "neutral_user": "Q\n (A) x\n (B) y\n\nAnswer:"})
    pressured = build_items([ex], pressured=True)[0]
    neutral = build_items([ex], pressured=False)[0]
    assert pressured.example.user.startswith("PRESSURE")
    assert neutral.example.user == ex.meta["neutral_user"]
    assert pressured.matching == neutral.matching == " (B)"
