"""Cross-type transfer cells from EXISTING probe.npz files (ADR 0011 step 5).

`compare.py` trains all types from scratch with default (leaky) designs; this
script instead takes probes you already trained on the ROLLOUT designs (the
trustworthy ones), builds each probe's own rollout dataset fresh, labels it
with the model (the mandatory filter), extracts activations once per distinct
dataset, and reports the full probe-x-dataset transfer AUROC matrix plus
direction cosines -- the headline transfer cells.

    # on OSC (7B, the model every probe was trained on):
    python -m scripts.crosstype --model Qwen/Qwen2.5-7B-Instruct \
        --probe syco=results/runs/<syco-run>/probe.npz \
        --probe sand-inst=results/runs/<instructed-run>/probe.npz \
        --probe sand-ince=results/runs/<incentive-run>/probe.npz@incentive

    # offline plumbing check (fixture, tiny, not trustworthy):
    python -m scripts.crosstype --source factual-small --probe ...

Probe spec: ``label=path[@pressure]``. `label` names the matrix row/column;
`path` is the probe.npz; the optional ``@incentive`` tells a sandbagging
probe's dataset which pressure arm to roll out under (default instructed).
Datasets are cached per (type, pressure), so two probes of the same arm share
one rollout+extraction pass. Evaluation uses the TEST split: no probe ever
sees its own training questions, and the shared mcq split keeps the boundary
identical across types.
"""

import argparse

import numpy as np

from dprobe import data, runlog
from dprobe.activations import extract, load_model, verify_read_positions
from dprobe.config import MODEL_NAME, SEED
from dprobe.evaluate import direction_cosines, transfer_matrix
from dprobe.plotting import report_comparison
from dprobe.steer import load_probe


def parse_spec(spec: str):
    """'label=path[@pressure]' -> (label, path, pressure)."""
    label, _, rest = spec.partition("=")
    if not rest:
        raise argparse.ArgumentTypeError(
            f"probe spec {spec!r} must be label=path[@pressure]")
    path, _, pressure = rest.partition("@")
    return label, path, (pressure or "instructed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="append", required=True, metavar="SPEC",
                    help="label=path/to/probe.npz[@pressure]; repeat per probe")
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id; MUST be the model every probe was "
                         "trained on (directions are model-specific).")
    ap.add_argument("--source", default="factual", choices=["factual", "factual-small"])
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="evaluation split; default 'test' (shared boundary across "
                         "types, so no probe is scored on its training questions)")
    ap.add_argument("--parse", default="strict", choices=["strict", "lenient"],
                    help="rollout answer parsing for the sandbagging engine")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap each dataset after filtering (speed)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    specs = [parse_spec(s) for s in args.probe]
    probes = {}
    for label, path, pressure in specs:
        p = load_probe(path)
        probes[label] = (p, pressure)
        print(f"probe {label}: type={p.deception_type} method={p.method} "
              f"layer={p.layer} (dataset pressure={pressure})")

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    from dprobe.data import rollout as _ro
    _ro.PARSE_MODE = args.parse

    # one labeled dataset + extraction per distinct (type, pressure)
    acts_cache: dict[tuple, tuple] = {}
    acts_by_label, labels_by_label = {}, {}
    for label, (p, pressure) in probes.items():
        key = (p.deception_type, pressure if p.deception_type == "sandbagging" else "-")
        if key not in acts_cache:
            t = p.deception_type
            print(f"building {key} dataset ({args.source}, {args.split} split) ...")
            examples = data.get(t, design="rollout", source=args.source,
                                split=args.split, pressure=pressure)
            flt = data.FILTERS[t]
            examples = flt(model, tokenizer, device, examples)
            print(f"  filter kept {len(examples)} examples")
            if args.max_examples is not None:
                examples = data.subsample(examples, args.max_examples, SEED)
            if not examples:
                raise SystemExit(f"dataset {key} is empty after the filter -- "
                                 "cannot fill this transfer cell")
            verify_read_positions(tokenizer, examples)
            acts, labels = extract(model, tokenizer, examples, device,
                                   batch_size=args.batch_size)
            acts_cache[key] = (acts, labels, len(examples))
        acts_by_label[label] = acts_cache[key][0]
        labels_by_label[label] = acts_cache[key][1]

    M, order = transfer_matrix({k: v[0] for k, v in probes.items()},
                               acts_by_label, labels_by_label)
    cos, _ = direction_cosines({k: v[0] for k, v in probes.items()})

    print("\nTransfer AUROC (row = probe, col = dataset):")
    header = "            " + "  ".join(f"{t:>10}" for t in order)
    print(header)
    for i, t in enumerate(order):
        print(f"{t:>10}  " + "  ".join(f"{M[i, j]:10.3f}" for j in range(len(order))))
    print("\nDirection cosines (NOTE: probes at different layers live in "
          "different spaces; cosine is only strictly meaningful same-layer):")
    for i, t in enumerate(order):
        print(f"{t:>10}  " + "  ".join(f"{cos[i, j]:10.3f}" for j in range(len(order))))

    run_dir = runlog.new_run_dir("crosstype", "mixed")
    report_comparison(M, cos, order, "crosstype", out_dir=run_dir)
    runlog.write_meta(run_dir, {
        "kind": "crosstype",
        "probes": {label: {"path": path, "pressure": pr,
                           "type": probes[label][0].deception_type,
                           "method": probes[label][0].method,
                           "layer": int(probes[label][0].layer)}
                   for label, path, pr in specs},
        "source": args.source,
        "split": args.split,
        "parse": args.parse,
        "seed": SEED,
        "model": model_name,
        "device": device,
        "dataset_sizes": {str(k): int(v[2]) for k, v in acts_cache.items()},
        "transfer_order": order,
        "transfer_auroc": [[float(x) for x in row] for row in M],
        "direction_cosines": [[float(x) for x in row] for row in cos],
    })
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
