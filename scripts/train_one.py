"""Train a probe for ONE deception type and write its diagnostic report.

    python -m scripts.train_one --type sycophancy
    python -m scripts.train_one --type sandbagging --method mms --filter
"""

import argparse

from dprobe import data
from dprobe.activations import extract, load_model
from dprobe.config import DECEPTION_TYPES, SEED
from dprobe.evaluate import layer_sweep
from dprobe.plotting import report_one_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=DECEPTION_TYPES)
    ap.add_argument("--method", default="lr", choices=["lr", "mms"])
    ap.add_argument("--filter", action="store_true",
                    help="sandbagging only: drop questions the model can't answer honestly")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap dataset size for speed; randomly drops whole prompts "
                         "(keeps matched pairs and label balance)")
    args = ap.parse_args()

    model, tokenizer, device = load_model()
    print(f"device: {device}")

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
    acts, labels = extract(model, tokenizer, examples, device)

    # hold out whole prompts: a pair's two halves share `user`, so this keeps
    # them on the same side of the split (genuinely independent held-out set)
    groups = [ex.user for ex in examples]
    aurocs, best_layer, probe = layer_sweep(acts, labels, args.method, args.type, groups)
    print(f"best layer {best_layer} | held-out AUROC {aurocs[best_layer]:.3f}")

    out = report_one_type(acts, labels, aurocs, best_layer, probe, args.method)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
