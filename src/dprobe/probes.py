"""The two probes, wrapped so a trained probe is a portable object.

A probe is, at heart, one direction vector in activation space plus a threshold.
Keeping it as an object (not loose numpy) is what makes the transfer study clean:
you can train a probe on sandbagging and call .score() on sycophancy activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class Probe:
    direction: np.ndarray   # unit vector in hidden space, points toward "deceptive"
    bias: float             # decision boundary along that direction
    layer: int              # which layer it was trained on
    method: str             # "mms" or "lr"
    deception_type: str     # what it was trained on, e.g. "sandbagging"

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed deception score for each row of X [n, hidden]. Higher = more deceptive."""
        return X @ self.direction - self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > 0).astype(int)


def fit_mms(X: np.ndarray, y: np.ndarray, layer: int, deception_type: str) -> Probe:
    """Difference-of-means probe. direction = mean(deceptive) - mean(honest)."""
    pos = X[y == 1].mean(0)
    neg = X[y == 0].mean(0)
    direction = pos - neg
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    bias = float((pos @ direction + neg @ direction) / 2)
    return Probe(direction, bias, layer, "mms", deception_type)


def fit_lr(X: np.ndarray, y: np.ndarray, layer: int, deception_type: str, C: float = 1.0) -> Probe:
    """Logistic-regression probe. Its weight vector becomes the direction."""
    clf = LogisticRegression(max_iter=2000, C=C).fit(X, y)
    w = clf.coef_[0]
    norm = np.linalg.norm(w) + 1e-8
    direction = w / norm
    # fold the sklearn intercept into the projected-space bias
    bias = float(-clf.intercept_[0] / norm)
    return Probe(direction, bias, layer, "lr", deception_type)


FITTERS = {"mms": fit_mms, "lr": fit_lr}
