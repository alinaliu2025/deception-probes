"""Cross-type transfer for DiD probes (ADR 0012), the difference-of-differences
analogue of scripts/crosstype.py.

Why a separate script: crosstype.py builds ROLLOUT datasets and scores probes on
raw *pressured-snapshot* activations. A did probe does NOT live in that space --
it is trained on ARROWS (pressured-minus-calm). Scoring a did direction on a raw
snapshot is a feature-space mismatch (ADR 0012: "transfer to types without a calm
baseline is undefined; the matrix needs a did-specific pairing on BOTH sides").
So here every dataset is that type's own did ARROWS, and each probe is read at its
own layer out of the target type's arrows -- apples to apples.

    # on OSC (7B, the model every probe was trained on):
    python -m scripts.crosstype_did --model Qwen/Qwen2.5-7B-Instruct \
        --probe syco=results/runs/<syco-did-run>/probe_promptfinal.npz \
        --probe sand-inst=results/runs/<sand-inst-did-run>/probe_promptfinal.npz

    # offline plumbing check (fixture, tiny, NOT trustworthy AUROC):
    python -m scripts.crosstype_did --source factual-small --probe ...

Probe spec: ``label=path[@pressure]``. `label` names the matrix row/column; `path`
is a did probe.npz (use the SAME read position for every probe -- default
prompt-final, the clean arm); the optional ``@incentive`` selects which sandbagging
pressure arm to build that probe's dataset under (default instructed; ignored for
sycophancy). Datasets are cached per (type, pressure), so probes of the same arm
share one gate+extraction pass. Evaluation uses the TEST split (shared mcq
boundary across types -> no probe is scored on its own training questions).
"""

import argparse

import numpy as np

from dprobe import data, runlog
from dprobe.activations import load_model
from dprobe.config import MODEL_NAME, SEED
from dprobe.did import extract_arrows
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
                    help="label=path/to/probe_promptfinal.npz[@pressure]; repeat per probe")
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id; MUST be the model every probe was "
                         "trained on (directions are model-specific).")
    ap.add_argument("--source", default="factual", choices=["factual", "factual-small"])
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="evaluation split; default 'test' (shared boundary across "
                         "types, so no probe is scored on its training questions)")
    ap.add_argument("--position", default="promptfinal",
                    choices=["promptfinal", "answertoken"],
                    help="which did read position's arrows to score on. MUST match "
                         "the position the probes were trained at (probe_promptfinal "
                         "-> promptfinal). Prompt-final is the clean arm (ADR 0012).")
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
    print(f"model: {model_name} | device: {device} | position: {args.position}")

    # one labeled dataset + arrow extraction per distinct (type, pressure). The
    # arrows[position] block is the feature space the did probe was trained in.
    acts_cache: dict[tuple, tuple] = {}
    acts_by_label, labels_by_label = {}, {}
    for label, (p, pressure) in probes.items():
        t = p.deception_type
        key = (t, pressure if t == "sandbagging" else "-")
        if key not in acts_cache:
            print(f"building {key} did dataset ({args.source}, {args.split} split) ...")
            examples = data.get(t, design="did", source=args.source,
                                split=args.split, pressure=pressure)
            # cap BEFORE the filter (like train_one) so --max-examples actually
            # bounds the expensive belief gate, not just the final example count
            if args.max_examples is not None:
                examples = data.subsample(examples, args.max_examples, SEED)
                print(f"  capped to {len(examples)} questions before gating")
            flt = data.FILTERS[t]              # routes did examples to did_filter
            examples = flt(model, tokenizer, device, examples)
            print(f"  did filter kept {len(examples)} labeled examples")
            if not examples:
                raise SystemExit(f"dataset {key} is empty after the did filter -- "
                                 "cannot fill this transfer cell (thin arm? scale up)")
            arrows, labels, _ = extract_arrows(
                model, tokenizer, device, examples, batch_size=args.batch_size)
            acts_cache[key] = (arrows[args.position], labels, len(examples))
        acts_by_label[label] = acts_cache[key][0]
        labels_by_label[label] = acts_cache[key][1]

    bare = {k: v[0] for k, v in probes.items()}
    M, order = transfer_matrix(bare, acts_by_label, labels_by_label)
    cos, _ = direction_cosines(bare)

    print("\nDiD transfer AUROC (row = probe, col = dataset's arrows):")
    print("            " + "  ".join(f"{t:>10}" for t in order))
    for i, t in enumerate(order):
        print(f"{t:>10}  " + "  ".join(f"{M[i, j]:10.3f}" for j in range(len(order))))
    print("\nDirection cosines (NOTE: probes at different layers live in different "
          "spaces; cosine is only strictly meaningful same-layer -- lead with AUROC):")
    for i, t in enumerate(order):
        print(f"{t:>10}  " + "  ".join(f"{cos[i, j]:10.3f}" for j in range(len(order))))

    run_dir = runlog.new_run_dir("crosstype_did", "mixed")
    report_comparison(M, cos, order, "crosstype_did", out_dir=run_dir)
    runlog.write_meta(run_dir, {
        "kind": "crosstype_did",
        "design": "did",
        "position": args.position,
        "probes": {label: {"path": path, "pressure": pr,
                           "type": probes[label][0].deception_type,
                           "method": probes[label][0].method,
                           "layer": int(probes[label][0].layer)}
                   for label, path, pr in specs},
        "source": args.source,
        "split": args.split,
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
