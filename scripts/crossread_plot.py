"""Turn a crossread run's meta.json into the figure + verdict for the write-up.

    python -m scripts.crossread_plot results/runs/<crossread run dir>

Reads ONLY meta.json (the one artifact git carries across machines), so this
runs on your laptop against a run produced on OSC. Writes crossread_plot.png
into the run dir and prints a plain-language verdict per region of the stack.

How to read the figure: solid lines are the diagonals (each read judged by its
own probe), dashed lines are the cross cells (each read judged by the OTHER
read's probe). The question is whether dashed tracks solid. The shaded band is
the "transfer gap" -- solid minus the worse dashed -- small gap = the two reads
expose one shared signal there; big gap = read-specific artifact territory.
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("kind") != "crossread":
        raise SystemExit(f"{run_dir} is a {meta.get('kind')!r} run, not crossread")
    return meta


def verdict(meta: dict) -> str:
    """Plain-language per-region readout of the 2x2 curves."""
    cc, tt = np.array(meta["auroc_cc"]), np.array(meta["auroc_tt"])
    ct, tc = np.array(meta["auroc_ct"]), np.array(meta["auroc_tc"])
    diag = np.minimum(cc, tt)
    cross = np.minimum(ct, tc)
    gap = diag - cross
    n = len(cc)
    lines = []
    lines.append(f"n_examples={meta['n_examples']} "
                 f"(test rows={meta['n_test_rows']}; AUROC resolution is coarse "
                 f"below ~100 test rows -- treat small gaps as noise)")
    for name, lo, hi in (("early (L1-5)", 1, min(6, n)),
                         ("mid", min(6, n), max(n - 6, 6)),
                         ("late", max(n - 6, 6), n)):
        d, c, g = diag[lo:hi].max(), cross[lo:hi].max(), gap[lo:hi].mean()
        if d < 0.65:
            call = "no usable signal in either read"
        elif c > d - 0.05:
            call = "SHARED signal: cross tracks diagonal -- read choice doesn't matter here"
        elif c < 0.6:
            call = "READ-SPECIFIC: strong diagonal but cross near chance -- artifact territory"
        else:
            call = "partial transfer: some shared signal plus a read-specific component"
        lines.append(f"{name:>12}: best diag {d:.3f} | best cross {c:.3f} "
                     f"| mean gap {g:+.3f}  -> {call}")
    bc, bt = meta["best_commit_layer"], meta["best_text_layer"]
    lines.append(f"best layers: commit L{bc} (C->C {cc[bc]:.3f}, C->T {ct[bc]:.3f}) | "
                 f"text L{bt} (T->T {tt[bt]:.3f}, T->C {tc[bt]:.3f})")
    return "\n".join(lines)


def plot(meta: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cc, tt = np.array(meta["auroc_cc"]), np.array(meta["auroc_tt"])
    ct, tc = np.array(meta["auroc_ct"]), np.array(meta["auroc_tc"])
    L = np.arange(len(cc))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(L, cc, "-", lw=2, label="C→C (commit probe on commit)")
    ax.plot(L, tt, "-", lw=2, label="T→T (text probe on text)")
    ax.plot(L, ct, "--", lw=2, label="C→T (commit probe on TEXT)")
    ax.plot(L, tc, "--", lw=2, label="T→C (text probe on COMMIT)")
    diag = np.minimum(cc, tt)
    cross = np.minimum(ct, tc)
    ax.fill_between(L, cross, diag, where=diag > cross, alpha=0.15,
                    label="transfer gap (diag − worse cross)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"cross-read 2×2 — {meta['type']} {meta['method']} "
                 f"{meta['source']} n={meta['n_examples']} ({meta['model']})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"figure -> {out_path}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    meta = load(run_dir)
    print(verdict(meta))
    plot(meta, run_dir / "crossread_plot.png")


if __name__ == "__main__":
    main()
