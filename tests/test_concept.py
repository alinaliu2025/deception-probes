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
