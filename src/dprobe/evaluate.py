"""Everything that turns activations + probes into numbers you can put in a slide.

The headline comparison is transfer_matrix: train a probe on deception type A,
measure how well it detects type B. A strong off-diagonal means the three
behaviours share a direction; a weak one means they're geometrically distinct.
This is the single-turn analogue of the LongHorizonDeception transfer question.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import SEED, TEST_FRAC
from .probes import FITTERS, Probe


def split(n: int, labels: np.ndarray):
    idx = np.arange(n)
    return train_test_split(idx, test_size=TEST_FRAC, random_state=SEED, stratify=labels)


def layer_sweep(acts: np.ndarray, labels: np.ndarray, method: str, deception_type: str):
    """Fit a probe at every layer, return (auroc_per_layer, best_layer, best_probe).

    AUROC is computed on a held-out split so the best layer isn't cherry-picked
    on training data.
    """
    n_layers = acts.shape[1]
    tr, te = split(len(labels), labels)
    aurocs = []
    best = {"layer": -1, "auroc": -1.0, "probe": None}
    for layer in range(n_layers):
        X = acts[:, layer, :]
        probe = FITTERS[method](X[tr], labels[tr], layer, deception_type)
        auroc = roc_auc_score(labels[te], probe.score(X[te]))
        aurocs.append(auroc)
        if auroc > best["auroc"]:
            best = {"layer": layer, "auroc": auroc, "probe": probe}
    # refit the winning probe on ALL data so it's the strongest version for transfer
    bl = best["layer"]
    best_probe = FITTERS[method](acts[:, bl, :], labels, bl, deception_type)
    return np.array(aurocs), bl, best_probe


def transfer_matrix(probes: dict[str, Probe], acts: dict[str, np.ndarray],
                    labels: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """AUROC of every probe applied to every dataset.

    probes[t] / acts[t] / labels[t] are keyed by deception type. Each probe carries
    its own layer, so we read that layer out of the target dataset's activations.
    Returns (matrix, types) where matrix[i, j] = probe_i evaluated on data_j.
    """
    types = list(probes.keys())
    M = np.zeros((len(types), len(types)))
    for i, ti in enumerate(types):
        p = probes[ti]
        for j, tj in enumerate(types):
            X = acts[tj][:, p.layer, :]
            M[i, j] = roc_auc_score(labels[tj], p.score(X))
    return M, types


def direction_cosines(probes: dict[str, Probe]) -> tuple[np.ndarray, list[str]]:
    """Cosine similarity between the learned directions. ~1 = same axis, ~0 = orthogonal."""
    types = list(probes.keys())
    D = np.stack([probes[t].direction for t in types])
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    return D @ D.T, types
