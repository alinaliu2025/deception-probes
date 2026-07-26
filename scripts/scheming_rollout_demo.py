"""Pass-1 demo: run the scheming rollout on a LOCAL reasoning model, read the
hidden state at the commit token, sweep layers.

A few reconstructed scenarios only. This is a plumbing check, not a trustworthy
AUROC. The model MUST be open-weights (you need the residual stream):

    python -m scripts.scheming_rollout_demo \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""

import argparse

import numpy as np

from dprobe import data
from dprobe.activations import extract, load_model, verify_read_positions
from dprobe.data import scheming
from dprobe.evaluate import layer_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    help="LOCAL open-weights reasoning model (o1 won't work: no acts)")
    ap.add_argument("--method", default="lr", choices=["lr", "mms", "mms_std", "lda"])
    ap.add_argument("--rollouts", type=int, default=None,
                    help="samples per scenario under pressure (default 6)")
    ap.add_argument("--prefix", default=None, choices=["reasoned", "commit"],
                    help="read after the model's own CoT, or on the bare '(A)'")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the calm gate (keep scenarios that scheme by default)")
    args = ap.parse_args()

    if args.rollouts is not None:
        scheming.SCHEME_ROLLOUT_N = args.rollouts
    if args.prefix is not None:
        scheming.SCHEME_PREFIX_MODE = args.prefix
    if args.no_gate:
        scheming.SCHEME_GATE = False

    model, tok, device = load_model(args.model)
    print(f"model: {args.model} | device: {device}")

    examples = data.get("scheming", design="rollout")
    print(f"built {len(examples)} pressured scenarios")

    labeled = scheming.scheming_rollout_filter(model, tok, device, examples)
    if not labeled:
        print("no ambivalent scenarios: widen the scenario set or raise --rollouts")
        return
    n1 = sum(e.label == 1 for e in labeled)
    print(f"labeled rollouts: {len(labeled)} ({n1} schemed / {len(labeled) - n1} straight)")

    verify_read_positions(tok, labeled)
    acts, labels = extract(model, tok, labeled, device)
    groups = [e.user for e in labeled]
    aurocs, best, _ = layer_sweep(acts, labels, args.method, "scheming", groups)
    print(f"best layer {best} | held-out AUROC {aurocs[best]:.3f} "
          f"(plumbing check on reconstructed data, not trustworthy)")


if __name__ == "__main__":
    main()
