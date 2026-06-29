"""Train a probe for ONE deception type and write its diagnostic report.

    python -m scripts.train_one --type sycophancy
    python -m scripts.train_one --type sandbagging --method mms --filter
"""

import argparse

import numpy as np

from dprobe import data, runlog
from dprobe.activations import extract, load_model
from dprobe.config import DECEPTION_TYPES, MODEL_NAME, SEED
from dprobe.evaluate import layer_sweep
from dprobe.plotting import report_one_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=DECEPTION_TYPES)
    ap.add_argument("--method", default="lr", choices=["lr", "mms", "mms_std", "lda"])
    ap.add_argument("--filter", action="store_true",
                    help="sandbagging only: drop questions the model can't answer honestly")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap dataset size for speed; randomly drops whole prompts "
                         "(keeps matched pairs and label balance)")
    ap.add_argument("--permute", action="store_true",
                    help="permutation control: shuffle labels before training. A "
                         "leak-free pipeline collapses to AUROC ~0.5; anything well "
                         "above means the held-out split is leaking signal.")
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

    if args.C is not None and args.method != "lr":
        print(f"warning: --C is lr-only and is ignored for --method {args.method}")

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    examples = data.get(args.type)
    if args.max_examples is not None:
        before = len(examples)
        examples = data.subsample(examples, args.max_examples, SEED)
        if len(examples) != before:
            print(f"capped to {len(examples)}/{before} examples (--max-examples {args.max_examples})")
    if args.type == "sandbagging" and args.filter:
        from dprobe.data.sandbagging import capability_filter
        before = len(examples)
        examples = capability_filter(model, tokenizer, device, examples)
        print(f"capability filter: kept {len(examples)}/{before} examples")

    print(f"extracting activations for {len(examples)} examples ...")
    acts, labels = extract(model, tokenizer, examples, device, batch_size=args.batch_size)

    if args.permute:
        labels = np.random.default_rng(SEED).permutation(labels)
        print("PERMUTATION CONTROL: labels shuffled -- a leak-free pipeline gives AUROC ~0.5")

    # hold out whole prompts: a pair's two halves share `user`, so this keeps
    # them on the same side of the split (genuinely independent held-out set)
    groups = [ex.user for ex in examples]
    aurocs, best_layer, probe = layer_sweep(acts, labels, args.method, args.type, groups, C=args.C)
    print(f"best layer {best_layer} | held-out AUROC {aurocs[best_layer]:.3f}")

    # immutable, self-describing run dir; meta.json is the part tracked in git
    run_dir = runlog.new_run_dir("permctrl" if args.permute else args.type, args.method)
    out = report_one_type(acts, labels, aurocs, best_layer, probe, args.method, groups, run_dir)
    np.savez(run_dir / "probe.npz", direction=probe.direction, bias=probe.bias,
             layer=probe.layer, method=probe.method, deception_type=probe.deception_type)
    np.save(run_dir / "aurocs.npy", aurocs)
    runlog.write_meta(run_dir, {
        "kind": "train_one",
        "type": args.type,
        "method": args.method,
        "seed": SEED,
        "max_examples": args.max_examples,
        "filter": bool(args.type == "sandbagging" and args.filter),
        "permuted": args.permute,
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "n_examples": int(len(labels)),
        "n_groups": int(len(set(groups))),
        "best_layer": int(best_layer),
        "auroc": float(aurocs[best_layer]),
        "aurocs": [float(a) for a in aurocs],
    })
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
