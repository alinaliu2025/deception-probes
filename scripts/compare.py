"""Train a probe for each deception type, then compare them.

This is the week's headline result:
  1. per-type probe (best layer, AUROC)
  2. transfer matrix: does a probe for type A detect type B?
  3. direction geometry: are the three directions shared or distinct?

    python -m scripts.compare --method lr
"""

import argparse

import numpy as np

from dprobe import data, runlog
from dprobe.activations import extract, load_model
from dprobe.config import DECEPTION_TYPES, MODEL_NAME, SEED
from dprobe.evaluate import direction_cosines, layer_sweep, transfer_matrix
from dprobe.plotting import report_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lr", choices=["lr", "mms", "mms_std", "lda"])
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id to probe; overrides config.MODEL_NAME "
                         "for this run only (e.g. Qwen/Qwen2.5-7B-Instruct). Default "
                         "keeps the committed baseline.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="extraction batch size; default auto-picks from GPU VRAM. "
                         "Lower it if a long sequence length OOMs.")
    ap.add_argument("--C", type=float, default=None,
                    help="lr only: inverse L2 strength. Lower (e.g. 0.1) converges "
                         "fast on near-separable data; default keeps sklearn C=1.0.")
    args = ap.parse_args()

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    acts, labels, probes, summary = {}, {}, {}, {}
    for t in DECEPTION_TYPES:
        examples = data.get(t)
        print(f"\n[{t}] extracting {len(examples)} examples ...")
        A, y = extract(model, tokenizer, examples, device, batch_size=args.batch_size)
        groups = [ex.user for ex in examples]
        aurocs, bl, probe = layer_sweep(A, y, args.method, t, groups, C=args.C)
        print(f"[{t}] best layer {bl} | AUROC {aurocs[bl]:.3f}")
        acts[t], labels[t], probes[t] = A, y, probe
        summary[t] = {"n_examples": int(len(y)), "n_groups": int(len(set(groups))),
                      "best_layer": int(bl), "auroc": float(aurocs[bl])}

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

    run_dir = runlog.new_run_dir("compare", args.method)
    report_comparison(M, cos, types, args.method, run_dir)
    np.save(run_dir / f"transfer_matrix_{args.method}.npy", M)
    runlog.write_meta(run_dir, {
        "kind": "compare",
        "method": args.method,
        "seed": SEED,
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "types": types,
        "per_type": summary,
        "transfer_matrix": M.tolist(),
        "direction_cosines": cos.tolist(),
    })
    print(f"\nsaved -> {run_dir}")


if __name__ == "__main__":
    main()
