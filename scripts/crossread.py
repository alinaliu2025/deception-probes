"""Cross-read comparison: does the probe signal survive changing WHERE we read?

    python -m scripts.crossread --type sycophancy --source factual-small --method mms
    python -m scripts.crossread --type sycophancy --source factual --method mms \
        --max-examples 1500 --model Qwen/Qwen2.5-7B-Instruct

Motivation (Jack's token-position question, 2026-07-10): the rollout pipeline
reads the hidden state on a RECONSTRUCTED prompt. Two reconstruction modes exist:

- commit  ("reenactment"): assistant prefix is the bare " (X)" -- kills the
  preamble-wording shortcut (ADR 0009 addendum), but the state is counterfactual
  (a model that answered instantly, with no lead-in).
- text    ("generation-time"): assistant prefix is the full sampled text cut at
  the "(X)". A forward pass over the same tokens reproduces the generation-time
  state, so this read IS (up to re-tokenization) the state at the moment the
  model actually committed -- but the wording confound is back in context.

Each read alone can look great (commit: mms L13 AUROC 1.0, 2026-07-06T11-07-48Z;
text: mms L15 0.98, 2026-07-07T02-21-08Z). What neither run shows is whether they
see the SAME signal. This script trains a probe on one read and evaluates it on
the other, per layer, on one shared rollout pass and one shared question split:

                          test on commit    test on text
    train on commit           C->C              C->T
    train on text             T->C              T->T

High off-diagonals = the two reads expose one shared direction (the read choice
was never the issue). Off-diagonals at chance while diagonals are high = each
read learns a read-specific artifact -- the token-position worry, made precise.
Expect the interesting case to be layer-dependent: the text read has an
early-layer wording signal the commit read lacks, so the cross cells may fail
early and hold mid-stack.

Mechanics: ONE rollout-filter pass in text mode (the expensive sampling happens
once). Every kept child carries the full sampled text as its prefix; its commit
twin is derived by swapping the prefix for the bare answer tag (which is
meta['matching'] iff label 1). Same questions, same labels, same grouped
train/test split on both sides -- the ONLY difference is the read. Note the
text-mode dedupe keys on (label, full text), so several children of one question
may share a commit twin; the pairing is kept 1:1 anyway so every cell scores the
identical example list.
"""

import argparse
from dataclasses import replace

import numpy as np

from dprobe import data, runlog
from dprobe.config import MODEL_NAME, SEED


def commit_twin(ex):
    """Derive the commit-read twin of a text-read rollout child: identical
    example, assistant prefix swapped to the bare answer tag of its label."""
    prefix = ex.meta["matching"] if ex.label == 1 else ex.meta["not_matching"]
    return replace(ex, meta={**ex.meta, "assistant_prefix": prefix})


def sweep_2x2(acts_c, acts_t, labels, groups, method, deception_type, C=None):
    """Per-layer 2x2: fit on each read's train rows, score both reads' test rows.

    One grouped split shared by every cell (the two activation sets are row-
    aligned twins, so the split transfers verbatim). Returns dict of four
    per-layer AUROC arrays keyed 'cc', 'ct', 'tt', 'tc'.
    """
    from sklearn.metrics import roc_auc_score

    from dprobe.evaluate import split
    from dprobe.probes import FITTERS

    fit_kwargs = {"C": C} if (method == "lr" and C is not None) else {}
    tr, te = split(len(labels), labels, np.asarray(groups))
    n_layers = acts_c.shape[1]
    curves = {k: [] for k in ("cc", "ct", "tt", "tc")}
    for layer in range(n_layers):
        Xc, Xt = acts_c[:, layer, :], acts_t[:, layer, :]
        pc = FITTERS[method](Xc[tr], labels[tr], layer, deception_type, **fit_kwargs)
        pt = FITTERS[method](Xt[tr], labels[tr], layer, deception_type, **fit_kwargs)
        curves["cc"].append(roc_auc_score(labels[te], pc.score(Xc[te])))
        curves["ct"].append(roc_auc_score(labels[te], pc.score(Xt[te])))
        curves["tt"].append(roc_auc_score(labels[te], pt.score(Xt[te])))
        curves["tc"].append(roc_auc_score(labels[te], pt.score(Xc[te])))
    return {k: np.array(v) for k, v in curves.items()}, (tr, te)


def render_table(curves) -> str:
    """Per-layer table of the four cells, with the transfer gap column."""
    lines = ["layer   C->C   C->T   T->T   T->C   min(cross)-0.5"]
    for L in range(len(curves["cc"])):
        cross = min(curves["ct"][L], curves["tc"][L])
        lines.append(f"{L:>5}  {curves['cc'][L]:.3f}  {curves['ct'][L]:.3f}  "
                     f"{curves['tt'][L]:.3f}  {curves['tc'][L]:.3f}   {cross - 0.5:+.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="sycophancy", choices=["sycophancy", "sandbagging"],
                    help="deception type; both rollout pipelines support the "
                         "text read, so both can be cross-read checked")
    ap.add_argument("--method", default="mms", choices=["lr", "mms", "mms_std", "lda"])
    ap.add_argument("--source", default="factual",
                    choices=["opinion", "factual", "factual-small"],
                    help="question source; 'factual-small' = offline smoke "
                         "(plumbing only, NOT trustworthy AUROC)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--gate", default="logprob", choices=["logprob", "sampled"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--acts-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--C", type=float, default=None, help="lr only")
    args = ap.parse_args()

    if args.type == "sandbagging" and args.source == "opinion":
        ap.error("sandbagging has no opinion source (ADR 0011); use factual/factual-small")

    # force TEXT prefix mode on the type's engine (the commit twin is derived);
    # sycophancy keeps its own knob module, sandbagging uses the shared engine
    if args.type == "sycophancy":
        from dprobe.data import sycophancy as knobs
    else:
        from dprobe.data import rollout as knobs
    knobs.ROLLOUT_PREFIX_MODE = "text"
    knobs.GATE_MODE = args.gate
    if args.rollouts is not None:
        knobs.ROLLOUT_N = args.rollouts
    if args.temperature is not None:
        knobs.ROLLOUT_TEMPERATURE = args.temperature
    if args.max_new_tokens is not None:
        knobs.ROLLOUT_MAX_NEW_TOKENS = args.max_new_tokens

    from dprobe.activations import extract, load_model, verify_read_positions

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    examples = data.get(args.type, design="rollout", source=args.source)
    if args.max_examples is not None:
        examples = data.subsample(examples, args.max_examples, SEED)
    flt = data.FILTERS[args.type]
    text_examples = flt(model, tokenizer, device, examples)
    if len(text_examples) < 8:
        raise SystemExit(f"only {len(text_examples)} examples survived the filter; "
                         "not enough for a split -- scale --max-examples up")
    filter_stats = getattr(flt, "last_stats", None)
    filter_log = getattr(flt, "last_log", None)

    commit_examples = [commit_twin(ex) for ex in text_examples]
    labels_expected = np.array([ex.label for ex in text_examples])
    groups = [ex.user for ex in text_examples]

    print("read-position check, TEXT read:")
    verify_read_positions(tokenizer, text_examples)
    print("read-position check, COMMIT read:")
    verify_read_positions(tokenizer, commit_examples)

    acts_dtype = np.dtype(args.acts_dtype)
    print(f"extracting TEXT-read activations for {len(text_examples)} examples ...")
    acts_t, labels = extract(model, tokenizer, text_examples, device,
                             batch_size=args.batch_size, acts_dtype=acts_dtype)
    print(f"extracting COMMIT-read activations for {len(commit_examples)} examples ...")
    acts_c, labels_c = extract(model, tokenizer, commit_examples, device,
                               batch_size=args.batch_size, acts_dtype=acts_dtype)
    assert (labels == labels_c).all() and (labels == labels_expected).all()

    curves, (tr, te) = sweep_2x2(acts_c, acts_t, labels, groups,
                                 args.method, args.type, C=args.C)

    best_c = int(np.argmax(curves["cc"]))
    best_t = int(np.argmax(curves["tt"]))
    print()
    print(render_table(curves))
    print()
    print(f"best commit layer L{best_c}: C->C {curves['cc'][best_c]:.3f} | "
          f"cross C->T {curves['ct'][best_c]:.3f}")
    print(f"best text   layer L{best_t}: T->T {curves['tt'][best_t]:.3f} | "
          f"cross T->C {curves['tc'][best_t]:.3f}")

    run_dir = runlog.new_run_dir("crossread", args.method)
    np.savez(run_dir / "crossread_curves.npz", **curves)
    if filter_log:
        (run_dir / "run_log.txt").write_text(filter_log, encoding="utf-8")
    (run_dir / "crossread_table.txt").write_text(render_table(curves), encoding="utf-8")
    runlog.write_meta(run_dir, {
        "kind": "crossread",
        "type": args.type,
        "method": args.method,
        "source": args.source,
        "gate": args.gate,
        "seed": SEED,
        "max_examples": args.max_examples,
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "acts_dtype": args.acts_dtype,
        "n_examples": int(len(labels)),
        "n_train_rows": int(len(tr)),
        "n_test_rows": int(len(te)),
        "n_groups": int(len(set(groups))),
        "filter_stats": filter_stats,
        "best_commit_layer": best_c,
        "best_text_layer": best_t,
        "auroc_cc": [float(a) for a in curves["cc"]],
        "auroc_ct": [float(a) for a in curves["ct"]],
        "auroc_tt": [float(a) for a in curves["tt"]],
        "auroc_tc": [float(a) for a in curves["tc"]],
    })
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
