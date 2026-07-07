"""Causally validate a probe direction by steering (ADR 0010).

Detection AUROC is correlational; this asks whether the direction *drives* caving.
Point it at a ``probe.npz`` from a train_one run and it runs two passes on fresh
sycophancy `factual` items:

  add     -- add alpha*v on the UNPRESSURED question, sweep alpha, report the
             wrong-answer rate (a real caving direction pushes it up).
  ablate  -- project v out on the PRESSURED prompt, report the caving-rate drop.

    # 7B on OSC, direction from a prior run you scp'd back:
    python -m scripts.steer --probe results/runs/<run>/probe.npz \
      --model Qwen/Qwen2.5-7B-Instruct --source factual

    # offline plumbing check (no network, not trustworthy):
    python -m scripts.steer --probe results/runs/<run>/probe.npz \
      --source factual-small

Writes its own run dir (kind="steer"); meta.json holds the curves.
"""

import argparse

import numpy as np

from dprobe import data, runlog
from dprobe.activations import load_model
from dprobe.config import MODEL_NAME, SEED
from dprobe.steer import add_sweep, ablate_pass, load_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    help="path to a probe.npz (direction + layer) from a train_one "
                         "or compare run. scp it back from OSC -- only meta.json is "
                         "git-tracked, the .npz stays on the machine that made it.")
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id; MUST match the model the probe was "
                         "trained on (a direction is model-specific). Default: config.")
    ap.add_argument("--source", default="factual", choices=["factual", "factual-small"],
                    help="question source. 'factual' = ARC (needs network); "
                         "'factual-small' = offline repo fixture for a plumbing check.")
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="which split to steer on; default 'test' (fresh items the "
                         "direction was not built from).")
    ap.add_argument("--mode", default="both", choices=["add", "ablate", "both"])
    ap.add_argument("--alphas", default="0,0.25,0.5,1,2,4",
                    help="comma-separated add-pass strengths. By default each is a "
                         "MULTIPLE of the layer's mean residual norm (portable "
                         "across model sizes); pass --raw to treat them as absolute.")
    ap.add_argument("--raw", action="store_true",
                    help="treat --alphas as absolute additions of alpha*unit_v "
                         "instead of scaling by the layer's residual norm.")
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap the number of questions (random subsample) for speed.")
    ap.add_argument("--samples", type=int, default=1,
                    help="completions per item: 1 = greedy (fast, deterministic); "
                         ">1 = sample at --temperature for a smoother rate.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature when --samples > 1.")
    ap.add_argument("--max-new-tokens", type=int, default=24,
                    help="generation cap per completion (must reach the '(X)').")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="generation batch size; default auto-picks from GPU VRAM.")
    args = ap.parse_args()

    probe = load_probe(args.probe)
    print(f"probe: {probe.deception_type} | method {probe.method} | layer {probe.layer}")

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    # raw, unlabeled factual examples (label -1); the filter is NOT run -- steering
    # selects its own baseline-correct / baseline-caved subsets by generation.
    examples = data.get("sycophancy", design="rollout", source=args.source,
                        split=args.split)
    if args.max_items is not None and len(examples) > args.max_items:
        examples = data.subsample(examples, args.max_items, SEED)
    print(f"steering on {len(examples)} {args.source} questions ({args.split} split)")

    alphas = [float(a) for a in args.alphas.split(",")]
    add_result = ablate_result = None
    if args.mode in ("add", "both"):
        print("ADD pass (elicit caving on unpressured items):")
        add_result = add_sweep(model, tokenizer, probe, examples, device, alphas,
                               raw=args.raw, samples=args.samples,
                               temperature=args.temperature,
                               max_new_tokens=args.max_new_tokens,
                               batch_size=args.batch_size)
    if args.mode in ("ablate", "both"):
        print("ABLATE pass (suppress caving on pressured items):")
        ablate_result = ablate_pass(model, tokenizer, probe, examples, device,
                                    samples=args.samples, temperature=args.temperature,
                                    max_new_tokens=args.max_new_tokens,
                                    batch_size=args.batch_size)

    run_dir = runlog.new_run_dir("steer", probe.method)
    if add_result is not None:
        np.save(run_dir / "add_curve.npy",
                np.array([add_result["alphas"], add_result["wrong_rate"]]))
    runlog.write_meta(run_dir, {
        "kind": "steer",
        "probe": str(args.probe),
        "deception_type": probe.deception_type,
        "probe_method": probe.method,
        "probe_layer": probe.layer,
        "source": args.source,
        "split": args.split,
        "mode": args.mode,
        "raw_alpha": args.raw,
        "samples": args.samples,
        "temperature": args.temperature,
        "seed": SEED,
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "add": add_result,
        "ablate": ablate_result,
    })
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
