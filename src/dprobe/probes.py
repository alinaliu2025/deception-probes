"""The two probes, wrapped so a trained probe is a portable object.

A probe is, at heart, one direction vector in activation space plus a threshold.
Keeping it as an object (not loose numpy) is what makes the transfer study clean:
you can train a probe on sandbagging and call .score() on sycophancy activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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


def fit_mms_std(X: np.ndarray, y: np.ndarray, layer: int, deception_type: str) -> Probe:
    """Standardized difference-of-means probe.

    Same estimator as ``fit_mms`` but computed in z-scored space: residual-stream
    dims have wildly different scales (a few massive-activation channels dominate
    the raw norm), so a raw mean-difference is swamped by those rogue dims rather
    than the deception signal. We take the mean-difference on standardized
    features, then fold the scaler back out so the stored direction stays in raw
    activation space and remains directly comparable across types for transfer --
    exactly the fold used in ``fit_lr``. Raw ``fit_mms`` is kept as-is; this is a
    separate method ("mms_std"), not a replacement.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    direction_z = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)
    # undo standardization: w_z·(x-mu)/sigma == (w_z/sigma)·x, so the raw-space
    # functional is w_z/sigma (the additive mu term cancels in a mean-difference)
    w = direction_z / scaler.scale_
    norm = np.linalg.norm(w) + 1e-8
    direction = w / norm
    proj = X @ direction
    bias = float((proj[y == 1].mean() + proj[y == 0].mean()) / 2)
    return Probe(direction, bias, layer, "mms_std", deception_type)


def fit_lr(X: np.ndarray, y: np.ndarray, layer: int, deception_type: str, C: float = 1.0) -> Probe:
    """Logistic-regression probe. Its weight vector becomes the direction.

    Residual-stream dimensions have wildly different scales, so we standardize
    before fitting: that is what lets lbfgs actually converge and makes the L2
    penalty hit every feature evenly. We then fold the scaler back into a single
    raw-activation linear functional, so the stored direction stays in raw space
    and remains directly comparable across types for the transfer study.
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=C).fit(scaler.transform(X), y)
    # undo standardization: w_s·(x-mu)/sigma + b_s == (w_s/sigma)·x + (b_s - (w_s/sigma)·mu)
    w = clf.coef_[0] / scaler.scale_
    intercept = float(clf.intercept_[0] - w @ scaler.mean_)
    norm = np.linalg.norm(w) + 1e-8
    direction = w / norm
    # fold the (raw-space) intercept into the projected-space bias
    bias = float(-intercept / norm)
    return Probe(direction, bias, layer, "lr", deception_type)


FITTERS = {"mms": fit_mms, "mms_std": fit_mms_std, "lr": fit_lr}
