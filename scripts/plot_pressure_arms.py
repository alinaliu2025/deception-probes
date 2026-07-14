"""Compare the two sandbagging pressure arms (ADR 0011 addendum) from meta.json.

Reads two train_one run metas (incentive + instructed) and writes three figures
into results/figures/ -- regenerable anytime from the git-tracked metas:

  pressure_arms_layer_sweep.png   per-layer held-out AUROC, both arms
  pressure_arms_yield.png         question funnel: in -> gated -> ambivalent -> examples
  pressure_arms_outcomes.png      question outcomes + unparsed-rollout share

Usage:
    python -m scripts.plot_pressure_arms \
        --incentive results/runs/<run>/meta.json \
        --instructed results/runs/<run>/meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"incentive": "#1f77b4", "instructed": "#d62728"}


def load(path: str) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    m["filter_stats"] = m.get("filter_stats") or {}
    return m


def label(m: dict) -> str:
    return f"{m['filter_stats'].get('pressure', '?')} (n={m['n_examples']})"


def plot_layer_sweep(arms: dict[str, dict], out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, m in arms.items():
        aurocs = m["aurocs"]
        ax.plot(range(len(aurocs)), aurocs, label=label(m), color=COLORS[arm], lw=2)
        bl = m["best_layer"]
        ax.scatter([bl], [aurocs[bl]], color=COLORS[arm], zorder=5)
        ax.annotate(f"L{bl}: {aurocs[bl]:.3f}", (bl, aurocs[bl]),
                    textcoords="offset points", xytext=(6, -12), fontsize=9,
                    color=COLORS[arm])
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("layer (residual stream)")
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("Sandbagging probe by layer: instructed vs incentive pressure\n"
                 f"(Qwen2.5-7B, ARC, within-question rollouts, method={arms['incentive']['method']})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "pressure_arms_layer_sweep.png", dpi=200)
    plt.close(fig)


def plot_yield(arms: dict[str, dict], out: Path):
    stages = ["questions\nin", "passed\ncapability gate", "ambivalent\nquestions",
              "final\nexamples"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    for k, (arm, m) in enumerate(arms.items()):
        fs = m["filter_stats"]
        vals = [fs["questions_in"], fs["sampled_questions"],
                fs["ambivalent_questions"], fs["n_examples"]]
        xs = [i + (k - 0.5) * width for i in range(len(stages))]
        bars = ax.bar(xs, vals, width, label=fs.get("pressure", arm),
                      color=COLORS[arm])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(100, 60000)  # headroom so the legend clears the bar labels
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("count (log scale)")
    ax.set_title("Yield funnel: the incentive-only prompt sandbagged rarely\n"
                 "(2.0% vs 24% of gated questions ambivalent)")
    ax.legend(title="pressure arm")
    fig.tight_layout()
    fig.savefig(out / "pressure_arms_yield.png", dpi=200)
    plt.close(fig)


def plot_outcomes(arms: dict[str, dict], out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2),
                                   gridspec_kw={"width_ratios": [2, 1]})
    # left: per-question outcomes, 100% stacked
    cats = [("ambivalent", "#2ca02c"), ("single-class", "#bbbbbb"),
            ("gate-failed", "#555555")]
    for k, (arm, m) in enumerate(arms.items()):
        fs = m["filter_stats"]
        total = fs["questions_in"]
        vals = [fs["ambivalent_questions"], fs["single_class_questions"],
                fs["dropped_gate_failed"]]
        bottom = 0.0
        for (name, color), v in zip(cats, vals):
            frac = v / total
            ax1.bar([k], [frac], bottom=bottom, color=color, width=0.6,
                    label=name if k == 0 else None)
            if frac > 0.04:
                ax1.text(k, bottom + frac / 2, f"{v:,}\n({frac:.1%})",
                         ha="center", va="center", fontsize=8)
            bottom += frac
        # small slices annotated above the bar
        if vals[0] / total <= 0.04:
            ax1.text(k, 1.02, f"ambivalent: {vals[0]:,} ({vals[0]/total:.1%})",
                     ha="center", fontsize=8, color=cats[0][1])
    ax1.set_xticks(range(len(arms)))
    ax1.set_xticklabels([m["filter_stats"].get("pressure", a)
                         for a, m in arms.items()])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("fraction of questions")
    ax1.set_title("Per-question outcome")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
               fontsize=8, frameon=False)

    # right: unparsed rollout share
    for k, (arm, m) in enumerate(arms.items()):
        fs = m["filter_stats"]
        n_roll = fs["sampled_questions"] * fs["n_rollouts"]
        frac = fs["unparsed_rollouts"] / n_roll
        ax2.bar([k], [frac], color=COLORS[arm], width=0.6)
        ax2.text(k, frac, f"{frac:.0%}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels([m["filter_stats"].get("pressure", a)
                         for a, m in arms.items()])
    ax2.set_ylim(0, 0.75)
    ax2.set_ylabel("unparsed rollouts")
    ax2.set_title("No '(X)' within token cap")
    fig.suptitle("Rollout outcomes by pressure arm (Qwen2.5-7B, 8 rollouts/question)")
    fig.tight_layout()
    fig.savefig(out / "pressure_arms_outcomes.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incentive", required=True, help="meta.json of the incentive run")
    ap.add_argument("--instructed", required=True, help="meta.json of the instructed run")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    arms = {"incentive": load(args.incentive), "instructed": load(args.instructed)}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_layer_sweep(arms, out)
    plot_yield(arms, out)
    plot_outcomes(arms, out)
    print(f"wrote 3 figures -> {out}/")


if __name__ == "__main__":
    main()
