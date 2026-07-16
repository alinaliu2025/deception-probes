"""One-figure dashboard of the 2026-07-13/14 sandbagging campaign.

Reads the git-tracked meta.json of the five runs (2 training, 2 steering,
permute, crosstype) and renders results/figures/sandbagging_dashboard.png:
transfer heatmap, steering add-curves vs random control, and a controls
scoreboard. Regenerable anytime:

    python -m scripts.plot_results_dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("results/runs")
R = {
    "train-ince": "2026-07-13T21-05-59Z_sandbagging_lr_115dd9d_p0316",
    "train-inst": "2026-07-13T22-54-02Z_sandbagging_lr_115dd9d_p0316",
    "steer-ince": "2026-07-14T16-43-21Z_steer_lr_b18954d_p0329",
    "steer-inst": "2026-07-14T16-43-35Z_steer_lr_b18954d_p0315",
    "permute":    "2026-07-14T19-15-59Z_permctrl_lr_b18954d_p0329",
    "crosstype":  "2026-07-14T19-56-35Z_crosstype_mixed_b18954d_p0329",
}
COL = {"inst": "#d62728", "ince": "#1f77b4"}


def meta(key):
    return json.loads((RUNS / R[key] / "meta.json").read_text())


def main():
    m = {k: meta(k) for k in R}
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.35)

    # --- A. transfer heatmap -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ct = m["crosstype"]
    order = ct["transfer_order"]
    M = np.array(ct["transfer_auroc"])
    sizes = ct["dataset_sizes"]
    n_by_col = {"syco": 84, "sand-inst": 108, "sand-ince": 16}  # from dataset_sizes
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3), [f"{t}\n(n={n_by_col[t]})" for t in order], fontsize=8)
    ax.set_yticks(range(3), order, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] < 0.7 else "black")
    ax.set_ylabel("probe trained on")
    ax.set_title("A. Transfer AUROC (test split)\nrow = probe, col = dataset",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- B/C. steering add curves -------------------------------------------
    for k, (arm, key, title) in enumerate([
            ("inst", "steer-inst", "B. Steering: instructed probe"),
            ("ince", "steer-ince", "C. Steering: incentive probe")]):
        ax = fig.add_subplot(gs[0, 1 + k])
        a = m[key]["add"]
        al = a["alphas"]
        ax.plot(al, a["wrong_rate"], "-o", color=COL[arm], label="probe direction")
        ctrl = (a.get("control") or {}).get("wrong_rate")
        if ctrl:
            ax.plot(al, ctrl, "-s", color="gray", label="random direction")
        ax.plot(al, a["parse_rate"], ":", color=COL[arm], alpha=0.5, label="parse rate")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("alpha (x residual norm)")
        ax.set_ylabel("wrong-answer rate")
        ax.set_title(f"{title}\nadd pass, {a['n_items']} baseline-correct items",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="center left")
        ab = m[key]["ablate"]
        ax.text(0.02, 0.72,
                f"ablate (n={ab['n_items']}): {ab['baseline_caving']:.2f} -> "
                f"{ab['ablated_caving']:.2f}\nrandom ctrl -> "
                f"{ab['control_ablated_caving']:.2f}",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    # --- D. detection summary ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    arms = ["instructed", "incentive"]
    aurocs = [m["train-inst"]["auroc"], m["train-ince"]["auroc"]]
    ns = [m["train-inst"]["n_examples"], m["train-ince"]["n_examples"]]
    bars = ax.bar(arms, aurocs, color=[COL["inst"], COL["ince"]], width=0.55)
    perm = m["permute"]["auroc"]
    ax.axhline(0.5, ls=":", color="gray", label="chance")
    ax.axhline(perm, ls="--", color="k",
               label=f"permutation ctrl: {perm:.2f} (instructed)")
    for b, a, n in zip(bars, aurocs, ns):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.015, f"{a:.3f}\n(n={n:,})",
                ha="center", fontsize=9)
    ax.set_ylim(0.4, 1.09)
    ax.set_ylabel("held-out AUROC")
    ax.set_title("D. Detection + leak control", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    # --- E. verdict table -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    rows = [
        ["check", "instructed arm", "incentive arm"],
        ["detection (held-out AUROC)", "0.993 (n=3,220)", "0.832 (n=156)"],
        ["permutation control", "0.56 ~ chance: no leak", "not run yet"],
        ["steering: add vs random", "0.02 vs 0.02 -- no effect", "0.01 vs 0.01 -- no effect"],
        ["steering: ablate vs random", "-0.02 = random", "n=7, no power"],
        ["transfer -> other arm", "0.64 (weak)", "0.37 (below chance!)"],
        ["transfer -> sycophancy", "0.95 (!)", "0.91 (!)"],
        ["syco probe -> this arm", "0.52 (none)", "0.52 (none)"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                   cellLoc="left", colWidths=[0.36, 0.33, 0.31])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    for j in range(3):
        tbl[0, j].set_facecolor("#e8e8e8")
        tbl[0, j].set_text_props(fontweight="bold")
    ax.set_title("E. Scoreboard: detection strong, causation absent,\n"
                 "transfer asymmetric (sandbagging probes see sycophancy, not vice versa)",
                 fontsize=10, fontweight="bold")

    fig.suptitle("Sandbagging probes: 2026-07-13/14 campaign (Qwen2.5-7B, ARC rollouts)",
                 fontsize=13, fontweight="bold")
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "sandbagging_dashboard.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'sandbagging_dashboard.png'}")


if __name__ == "__main__":
    main()
