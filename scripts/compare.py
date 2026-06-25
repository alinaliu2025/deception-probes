"""Train a probe for each deception type, then compare them.

This is the week's headline result:
  1. per-type probe (best layer, AUROC)
  2. transfer matrix: does a probe for type A detect type B?
  3. direction geometry: are the three directions shared or distinct?

    python -m scripts.compare --method lr
"""

import argparse

import numpy as np

from dprobe import data
from dprobe.activations import extract, load_model
from dprobe.config import DECEPTION_TYPES
from dprobe.evaluate import direction_cosines, layer_sweep, transfer_matrix
from dprobe.plotting import report_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lr", choices=["lr", "mms"])
    args = ap.parse_args()

    model, tokenizer, device = load_model()
    print(f"device: {device}")

    acts, labels, probes = {}, {}, {}
    for t in DECEPTION_TYPES:
        examples = data.get(t)
        print(f"\n[{t}] extracting {len(examples)} examples ...")
        A, y = extract(model, tokenizer, examples, device)
        aurocs, bl, probe = layer_sweep(A, y, args.method, t)
        print(f"[{t}] best layer {bl} | AUROC {aurocs[bl]:.3f}")
        acts[t], labels[t], probes[t] = A, y, probe

    M, types = transfer_matrix(probes, acts, labels)
    cos, _ = direction_cosines(probes)

    print("\ntransfer AUROC (row=train, col=test):")
    print("        " + "  ".join(f"{t[:6]:>6}" for t in types))
    for i, t in enumerate(types):
        print(f"{t[:6]:>6}  " + "  ".join(f"{M[i, j]:>6.2f}" for j in range(len(types))))

    print("\ndirection cosine similarity:")
    print("        " + "  ".join(f"{t[:6]:>6}" for t in types))
    for i, t in enumerate(types):
        print(f"{t[:6]:>6}  " + "  ".join(f"{cos[i, j]:>6.2f}" for j in range(len(types))))

    out = report_comparison(M, cos, types, args.method)
    print(f"\nsaved -> {out}")
    np.save(f"results/transfer_matrix_{args.method}.npy", M)


if __name__ == "__main__":
    main()
