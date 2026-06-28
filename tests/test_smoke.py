"""Fast checks that need no model download. Run:  pytest -q

These validate the math and the dataset contract, so you catch a broken refactor
in seconds instead of after a 5-minute activation extraction. The actual model
run is exercised by scripts/train_one.py.
"""

import numpy as np

from dprobe import data
from dprobe.config import DECEPTION_TYPES
from dprobe.evaluate import direction_cosines, layer_sweep, transfer_matrix
from dprobe.probes import fit_lr, fit_mms, fit_mms_std


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


def test_probes_recover_planted_direction():
    acts, y = _separable()
    X = acts[:, 1, :]
    for fitter in (fit_mms, fit_mms_std, fit_lr):
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
