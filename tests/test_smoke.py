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
