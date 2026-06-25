"""Figures. Two entry points: report_one_type (per-probe diagnostics) and
report_comparison (the cross-type story)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from .config import RESULTS_DIR
from .evaluate import split
from .probes import Probe

C_HONEST, C_DECEP = "#2a7fb8", "#c4452f"


def report_one_type(acts, labels, aurocs, best_layer, probe: Probe, method: str):
    """The 2x2 diagnostic from the original script, generalised to one deception type."""
    n_layers = acts.shape[1]
    tr, te = split(len(labels), labels)
    Xbl = acts[:, best_layer, :]
    scores = probe.score(Xbl)
    test_auroc = aurocs[best_layer]
    fpr, tpr, _ = roc_curve(labels[te], probe.score(Xbl[te]))
    pca = PCA(n_components=2).fit_transform(Xbl)

    honest, decep = labels == 0, labels == 1
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"{probe.deception_type} | {method} | layer {best_layer}, "
                 f"AUROC {test_auroc:.2f}", fontsize=13, fontweight="bold")

    a = ax[0, 0]
    a.plot(range(n_layers), aurocs, "-o", ms=3, color=C_DECEP)
    a.axvline(best_layer, ls="--", color="gray", lw=1)
    a.axhline(0.5, ls=":", color="gray", lw=1)
    a.set(xlabel="layer", ylabel="held-out AUROC", title="A. Where is it decodable?",
          ylim=(0.3, 1.05))

    b = ax[0, 1]
    bins = np.linspace(scores.min(), scores.max(), 16)
    b.hist(scores[honest], bins=bins, alpha=0.7, label="control", color=C_HONEST)
    b.hist(scores[decep], bins=bins, alpha=0.7, label="deceptive", color=C_DECEP)
    b.axvline(0, ls="--", color="gray", lw=1)
    b.set(xlabel="probe score", ylabel="count", title=f"B. Separation (layer {best_layer})")
    b.legend(fontsize=8)

    c = ax[1, 0]
    c.scatter(pca[honest, 0], pca[honest, 1], c=C_HONEST, label="control", alpha=0.8,
              edgecolor="w", s=45)
    c.scatter(pca[decep, 0], pca[decep, 1], c=C_DECEP, label="deceptive", alpha=0.8,
              edgecolor="w", s=45)
    c.set(xlabel="PC1", ylabel="PC2", title=f"C. Geometry (layer {best_layer})")
    c.legend(fontsize=8)

    d = ax[1, 1]
    d.plot(fpr, tpr, "-", color=C_DECEP, lw=2, label=f"AUROC = {test_auroc:.3f}")
    d.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
    d.set(xlabel="false positive rate", ylabel="true positive rate", title="D. ROC (held-out)")
    d.legend(fontsize=9, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS_DIR / f"report_{probe.deception_type}_{method}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _heatmap(ax, M, types, title, fmt="{:.2f}"):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)), types, rotation=20, ha="right")
    ax.set_yticks(range(len(types)), types)
    for i in range(len(types)):
        for j in range(len(types)):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="w" if M[i, j] < 0.7 else "black", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    return im


def report_comparison(transfer_M, cos_M, types, method: str):
    """Two heatmaps side by side: cross-type transfer AUROC and direction cosine."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = _heatmap(ax[0], transfer_M, types,
                   f"Transfer AUROC ({method})\nrow = trained on, col = tested on")
    ax[0].set_ylabel("probe trained on")
    ax[0].set_xlabel("evaluated on")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    # cosine can be negative; remap display range
    im1 = ax[1].imshow(cos_M, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(types)), types, rotation=20, ha="right")
    ax[1].set_yticks(range(len(types)), types)
    for i in range(len(types)):
        for j in range(len(types)):
            ax[1].text(j, i, f"{cos_M[i, j]:.2f}", ha="center", va="center",
                       color="black", fontsize=10)
    ax[1].set_title("Direction cosine similarity\n(1 = shared axis, 0 = orthogonal)",
                    fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    fig.suptitle(f"Cross-type deception probe comparison | {method}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / f"comparison_{method}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
