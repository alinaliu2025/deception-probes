"""Fast checks for the concept-vector pipeline (ADR 0014). No model download.

These cover the parts that are pure math or pure contract: the fixture, the judge
filter, the direction, and the sign convention. What they CANNOT cover is
`extract_response_mean`, which needs a real tokenizer and a real forward pass.
That one gets checked by eye on the 0.5B plumbing run (see TODO.md next action),
which is why its docstring carries the read-position warning.
"""

import json

import numpy as np
import pytest

from dprobe import concept
from dprobe.concept import ConceptSpec
from dprobe.probes import Probe


def _poled(n=80, hidden=32, layers=5, planted=3, seed=0):
    """Synthetic activations with a trait direction planted at one layer only."""
    rng = np.random.default_rng(seed)
    poles = np.array([0, 1] * (n // 2))
    acts = rng.normal(size=(n, layers, hidden))
    acts[poles == 1, planted, 0] += 5.0
    return acts, poles


# --- fixture ---------------------------------------------------------------

def test_fixture_loads_with_equal_poles():
    spec = ConceptSpec.load()
    assert spec.trait == "sandbagging"
    assert len(spec.positive_systems) == len(spec.negative_systems)
    assert len(spec.questions) >= 10
    assert spec.description and spec.judge_rubric


def test_fixture_rejects_unequal_poles(tmp_path):
    """Unequal poles put a prompt-count artifact into the difference of means."""
    spec = json.loads(concept.FIXTURE.read_text(encoding="utf-8"))
    spec.pop("_comment", None)
    spec["positive_systems"] = spec["positive_systems"][:2]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="pole imbalance"):
        ConceptSpec.load(p)


def test_questions_are_shared_across_poles():
    """Content confounding control: the same questions must appear on both poles,
    so topic cannot align with the pole. This is a property of the SPEC, not of
    the generation code, so it is worth asserting here."""
    spec = ConceptSpec.load()
    assert len(set(spec.questions)) == len(spec.questions), "duplicate questions"


# --- judge filter ----------------------------------------------------------

def _records(n_pos=30, n_neg=30):
    return ([{"pole": "pos", "system": "s", "user": "u", "response": "r"}] * n_pos
            + [{"pole": "neg", "system": "s", "user": "u", "response": "r"}] * n_neg)


def test_judge_keeps_only_the_clean_tails():
    spec = ConceptSpec.load()
    # 60 per pole, so the surviving half still clears the 20-per-pole floor
    recs = [dict(r) for r in _records(60, 60)]
    # alternate high/low scores so half of each pole should survive
    scores = iter([90, 10] * len(recs))
    kept = concept.judge_responses(recs, spec, lambda _p: next(scores),
                                   pos_min=50, neg_max=20)
    assert all(r["score"] >= 50 for r in kept if r["pole"] == "pos")
    assert all(r["score"] <= 20 for r in kept if r["pole"] == "neg")
    assert len(kept) < len(recs), "filter kept everything, thresholds are inert"


def test_judge_refuses_to_run_on_too_few_survivors():
    """Lowering the threshold to make a run go through is the failure mode this
    guard exists to block."""
    spec = ConceptSpec.load()
    recs = [dict(r) for r in _records()]
    with pytest.raises(ValueError, match="too few survivors"):
        concept.judge_responses(recs, spec, lambda _p: 35)  # nothing clears either tail


def test_judge_prompt_carries_rubric_and_response():
    spec = ConceptSpec.load()
    seen = {}
    recs = [dict(r) for r in _records()]
    for r in recs:
        r["response"] = "MARKER_TEXT"

    def score(prompt):
        seen["prompt"] = prompt
        return 90 if "MARKER_TEXT" in prompt else 0

    concept.judge_responses(recs, spec, score, pos_min=50, neg_max=95)
    assert spec.judge_rubric in seen["prompt"]
    assert "MARKER_TEXT" in seen["prompt"]


# --- capability pairing ----------------------------------------------------

def _paired(questions, n_per_pole=12):
    return [{"pole": p, "system": f"{p}-sys", "user": q, "response": f"{p}{k}"}
            for q in questions for p in ("pos", "neg")
            for k in range(n_per_pole)]


def _grader(control_correct: dict, pressured_correct: dict):
    """Fake correctness judge driven by two per-question lookup tables.

    Keys off the RESPONSE text, because `build_correctness_prompt` deliberately
    does not show the judge the system prompt. Telling a correctness grader which
    condition produced an answer would bias the grading, which is the confound
    the pairing exists to avoid."""
    def score(prompt: str) -> int:
        q = next(l for l in prompt.splitlines() if l.startswith("q"))
        resp = prompt.split("Assistant answer:\n")[1].split("\n")[0]
        table = control_correct if resp.startswith("neg") else pressured_correct
        return 100 if table.get(q, False) else 0
    return score


def test_capability_filter_drops_questions_the_model_never_knew():
    """A wrong answer from a model that never knew the answer is ignorance, not
    sandbagging. CLAUDE.md requires this filter for the sandbagging type."""
    spec = ConceptSpec.load()
    qs = ["q1", "q2", "q3", "q4"]
    recs = _paired(qs)
    # control fails q3 entirely -> q3 must be dropped for BOTH poles
    score = _grader({"q1": True, "q2": True, "q3": False, "q4": True}, {})
    kept = concept.capability_pair_filter(recs, spec, score, verbose=False)
    assert all(r["user"] != "q3" for r in kept)
    assert {r["user"] for r in kept} == {"q1", "q2", "q4"}


def test_poles_are_pressured_wrong_versus_control_right():
    spec = ConceptSpec.load()
    recs = _paired(["q1", "q2", "q3"])
    score = _grader({"q1": True, "q2": True, "q3": True}, {})
    kept = concept.capability_pair_filter(recs, spec, score, verbose=False)
    assert all(r["correct"] <= 0 for r in kept if r["pole"] == "pos")
    assert all(r["correct"] >= 100 for r in kept if r["pole"] == "neg")


def test_a_compliant_model_yields_no_positive_pole():
    """If the pressure systems do not work, every pressured answer is correct and
    the positive pole is empty. That must abort, not build a direction out of the
    handful that happened to be wrong."""
    spec = ConceptSpec.load()
    recs = _paired(["q1", "q2", "q3"])
    score = _grader({"q1": True, "q2": True, "q3": True},
                    {"q1": True, "q2": True, "q3": True})
    with pytest.raises(ValueError, match="too few survivors"):
        concept.capability_pair_filter(recs, spec, score, verbose=False)


def test_correctness_prompt_requires_a_rubric():
    spec = ConceptSpec.load()
    spec.correctness_rubric = ""
    with pytest.raises(ValueError, match="no `correctness_rubric`"):
        concept.build_correctness_prompt(spec, {"user": "u", "response": "r"})


def test_correctness_prompt_tells_the_judge_to_ignore_length():
    """The 0.5B run failed because the judge scored terseness. Both rubrics now
    say so explicitly; if that sentence is ever dropped, this fails."""
    spec = ConceptSpec.load()
    assert "not its length" in spec.correctness_rubric
    assert "Do not infer intent from brevity" in spec.judge_rubric


# --- direction -------------------------------------------------------------

def test_fit_concept_recovers_planted_direction():
    acts, poles = _poled()
    probe = concept.fit_concept(acts, poles, layer=3, deception_type="sandbagging")
    assert probe.method == "concept"
    assert probe.layer == 3
    assert np.isclose(np.linalg.norm(probe.direction), 1.0)
    assert abs(probe.direction[0]) > 0.9, "did not recover the planted axis"


def test_sign_convention_matches_behavioural_probes():
    """CLAUDE.md: 1 = deceptive, never flip. A concept probe must score the
    trait-expressing pole HIGHER, or its cosine with a behavioural probe reads
    backwards and every transfer number flips sign."""
    acts, poles = _poled()
    probe = concept.fit_concept(acts, poles, layer=3, deception_type="sandbagging")
    s = probe.score(acts[:, 3, :])
    assert s[poles == 1].mean() > s[poles == 0].mean()


def test_bias_puts_threshold_between_the_poles():
    acts, poles = _poled()
    probe = concept.fit_concept(acts, poles, layer=3, deception_type="sandbagging")
    pred = probe.predict(acts[:, 3, :])
    assert (pred == poles).mean() > 0.95


def test_layer_sweep_finds_the_planted_layer():
    acts, poles = _poled(planted=2)
    probes, seps = concept.layer_sweep_concept(acts, poles, "sandbagging")
    assert len(probes) == acts.shape[1]
    assert int(np.argmax(seps)) == 2


def test_layer_selection_is_not_fooled_by_growing_activation_norm():
    """Regression, 2026-07-27. Residual norms grow with depth in a transformer.
    A separation metric measured in raw units climbs with the layer index whether
    or not the layer carries signal, so argmax lands on the last layer every time.
    The first 0.5B run picked layer 24 of 24 that way.

    Here layer 4 has 50x the scale of every other layer and NO planted signal,
    while layer 2 has the signal. A scale-free metric picks 2."""
    acts, poles = _poled(planted=2)
    acts[:, 4, :] *= 50.0
    _, seps = concept.layer_sweep_concept(acts, poles, "sandbagging")
    assert int(np.argmax(seps)) == 2, (
        f"layer selection followed activation scale, not signal: {seps.round(2)}"
    )


def test_separation_is_invariant_to_rescaling_a_layer():
    """The same statement as a property: multiplying a layer by a constant must
    not change its separation score."""
    acts, poles = _poled(planted=2)
    _, before = concept.layer_sweep_concept(acts, poles, "sandbagging")
    acts[:, 2, :] *= 7.0
    _, after = concept.layer_sweep_concept(acts, poles, "sandbagging")
    assert np.isclose(before[2], after[2], rtol=1e-6)


def test_score_fixed_direction_does_not_refit():
    """The whole point of ADR 0014: score a direction that already exists.

    Fit on one dataset, score on an INDEPENDENT one. A real direction transfers; a
    random one sits at chance. If `score_fixed_direction` ever quietly refits, the
    random arm jumps and this fails."""
    acts_a, poles_a = _poled(seed=0)
    acts_b, poles_b = _poled(seed=1)

    good = concept.fit_concept(acts_a, poles_a, layer=3,
                               deception_type="sandbagging")
    assert concept.score_fixed_direction(acts_b, poles_b, good) > 0.95

    rng = np.random.default_rng(7)
    v = rng.normal(size=acts_a.shape[2])
    v /= np.linalg.norm(v)
    rand = Probe(v, 0.0, 3, "concept", "sandbagging")
    assert 0.3 < concept.score_fixed_direction(acts_b, poles_b, rand) < 0.7


def test_in_sample_diff_of_means_scores_well_above_chance_on_pure_noise():
    """Not a bug, a warning, and the reason the concept direction is only ever
    scored out-of-sample.

    Layer 0 has no planted signal. Fitting a difference of means there and scoring
    it on the SAME rows still clears chance by a wide margin, because 32 dimensions
    and 80 samples is enough room to fit noise. Every in-sample AUROC in this
    project has this floor under it."""
    acts, poles = _poled()
    overfit = concept.fit_concept(acts, poles, layer=0,
                                  deception_type="sandbagging")
    in_sample = concept.score_fixed_direction(acts, poles, overfit)
    assert in_sample > 0.7, "expected in-sample overfit on noise"

    acts_b, poles_b = _poled(seed=1)
    assert concept.score_fixed_direction(acts_b, poles_b, overfit) < 0.7
