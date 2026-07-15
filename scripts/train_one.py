"""Train a probe for ONE deception type and write its diagnostic report.

    python -m scripts.train_one --type sycophancy
    python -m scripts.train_one --type sandbagging --method mms --filter
"""

import argparse
from dataclasses import replace

import numpy as np

from dprobe import data, runlog
from dprobe.activations import extract, load_model, verify_read_positions
from dprobe.config import DECEPTION_TYPES, MODEL_NAME, SEED
from dprobe.evaluate import layer_sweep
from dprobe.plotting import report_one_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=DECEPTION_TYPES)
    ap.add_argument("--method", default="lr", choices=["lr", "mms", "mms_std", "lda"])
    ap.add_argument("--filter", action="store_true",
                    help="run the type's behaviour filter: sandbagging drops "
                         "questions the model can't answer honestly; sycophancy "
                         "framing drops questions where the framing doesn't flip "
                         "the model's answer; sycophancy behavioral ASSIGNS the "
                         "labels from the model's own choice (mandatory there)")
    ap.add_argument("--design", default="completion",
                    choices=["completion", "framing", "behavioral", "rollout", "did"],
                    help="sycophancy only: 'completion' (default) is the original "
                         "leaky answer-paste baseline; 'framing' is the instruction "
                         "contrast (leaks the instruction tokens, ADR 0007); "
                         "'behavioral' labels by the model's own choice under an "
                         "identical pressure prompt -- requires --filter, and is "
                         "content-confounded (ADR 0008); 'rollout' labels each "
                         "SAMPLED answer to the same question (within-question "
                         "contrast, ADR 0008, pending sign-off) -- requires --filter; "
                         "'did' is difference-of-differences (ADR 0012, pending "
                         "sign-off): per-question caved/held label, then the "
                         "pressured-minus-calm ARROW cancels question content -- "
                         "writes a probe+report for BOTH read positions "
                         "(promptfinal, answertoken) -- requires --filter.")
    ap.add_argument("--source", default="opinion",
                    choices=["opinion", "factual", "factual-small"],
                    help="sycophancy behavioral/rollout only: question source. "
                         "'opinion' (default) is the model-written-evals opinion "
                         "data; 'factual' recasts ARC Easy+Challenge MCQs with the "
                         "user asserting a WRONG answer, so the belief gate becomes "
                         "a capability check and ambivalence yield goes up "
                         "(ADR 0009, Proposed -- pending sign-off); 'factual-small' "
                         "is a tiny repo-resident OFFLINE stand-in for 'factual' "
                         "(fixtures/factual_smoke.jsonl) for smoke tests and demos "
                         "-- runs with no network, NOT for trustworthy AUROC.")
    ap.add_argument("--pressure", default="instructed",
                    choices=["instructed", "incentive"],
                    help="sandbagging did only: pressure system prompt. "
                         "'instructed' (default) demands a wrong answer; "
                         "'incentive' states only the evaluation incentive "
                         "(ADR 0011 addendum).")
    ap.add_argument("--rollouts", type=int, default=None,
                    help="rollout design only: samples per question (default 8)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="rollout design only: sampling temperature (default 1.0)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="rollout design only: generation cap per rollout (default "
                         "24). Too low truncates conversational answers before "
                         "their '(X)' -- they count as unparsed AND the kept set "
                         "skews to format-compliant rollouts (ADR 0009 addendum).")
    ap.add_argument("--gate", default=None, choices=["logprob", "sampled"],
                    help="sycophancy behavioral/rollout/did belief gate. 'logprob' "
                         "is the deterministic teacher-forced comparison (ADR 0007); "
                         "'sampled' generates --gate-n answers on the unpressured "
                         "prompt and keeps the question only if it answers correctly "
                         "on >= --gate-threshold of them (the v2-plan 'model is SURE' "
                         "gate; costs a generate call per question). Default is "
                         "'logprob', except --design did defaults to 'sampled' (the "
                         "consistency prerequisite, ADR 0012).")
    ap.add_argument("--gate-n", type=int, default=None,
                    help="--gate sampled only: samples per question (default 20)")
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help="--gate sampled only: min correct fraction to keep a "
                         "question (default 0.9)")
    ap.add_argument("--gate-temperature", type=float, default=None,
                    help="--gate sampled only: sampling temperature (default 0.7)")
    ap.add_argument("--rollout-prefix", default=None, choices=["commit", "text"],
                    help="rollout design only: what the read position sits on. "
                         "'commit' (default) = bare ' (X)' answer string, kills "
                         "the preamble-wording shortcut; 'text' = full sampled "
                         "text up to the '(X)' (original ADR 0008) -- the "
                         "ablation arm measuring that shortcut (ADR 0009 addendum).")
    ap.add_argument("--read-prompt", default="pressure", choices=["pressure", "neutral"],
                    help="sycophancy behavioral only: 'neutral' is the confound "
                         "control -- keep the filter-assigned labels but extract "
                         "activations on the persona-stripped NEUTRAL prompt, where "
                         "no pressure exists so nothing deceptive can be happening. "
                         "AUROC here measures pure question-content signal; the "
                         "pressure run is only trustworthy above this baseline.")
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
    ap.add_argument("--acts-dtype", default="float32", choices=["float32", "float16"],
                    help="dtype of the host-side activation buffer (the run's main "
                         "RAM cost). float16 halves it and is lossless on an fp16 "
                         "model; use it to fit large example counts on a small node.")
    ap.add_argument("--C", type=float, default=None,
                    help="lr only: inverse L2 strength. Lower (e.g. 0.1) converges "
                         "fast on near-separable data; default keeps sklearn C=1.0.")
    args = ap.parse_args()

    if args.C is not None and args.method != "lr":
        print(f"warning: --C is lr-only and is ignored for --method {args.method}")
    # which (type, design) pairs exist: sycophancy has all of them; sandbagging
    # has 'completion' (legacy seed) and 'did' (ADR 0012 ported, calm_system swap)
    _did_run = args.design == "did" and args.type in ("sycophancy", "sandbagging")
    if args.design != "completion" and args.type not in ("sycophancy", "sandbagging"):
        print(f"warning: --design is ignored for --type {args.type}")
    if args.type == "sandbagging" and args.design not in ("completion", "did"):
        ap.error(f"--design {args.design} is not defined for sandbagging "
                 "(have 'completion', 'did')")
    if args.pressure != "instructed" and not (
            args.type == "sandbagging" and args.design == "did"):
        ap.error("--pressure incentive only applies to --type sandbagging "
                 "--design did (ADR 0011 addendum)")
    if ((args.type == "sycophancy" and args.design in ("behavioral", "rollout", "did"))
            or (args.type == "sandbagging" and args.design == "did")) and not args.filter:
        ap.error(f"--design {args.design} requires --filter: labels are assigned by "
                 "running the model (build emits the -1 sentinel only)")
    if args.source != "opinion" and not (
            args.type == "sycophancy" and args.design in ("behavioral", "rollout", "did")
            or (args.type == "sandbagging" and args.design == "did")):
        ap.error(f"--source {args.source} requires sycophancy --design behavioral/"
                 "rollout/did or sandbagging --design did (ADR 0009/0012)")
    if args.type == "sandbagging" and args.design == "did" and args.source == "opinion":
        ap.error("sandbagging --design did needs --source factual (ARC) or "
                 "factual-small (offline fixture); there is no opinion source "
                 "for sandbagging")
    if (args.rollouts is not None or args.temperature is not None
            or args.rollout_prefix is not None) and not (
            args.type == "sycophancy" and args.design == "rollout"):
        ap.error("--rollouts/--temperature/--rollout-prefix "
                 "only apply to --type sycophancy --design rollout")
    # --max-new-tokens caps generation length; it applies to both designs that
    # generate (rollout, and did -- its belief gate and pressured answer).
    if args.max_new_tokens is not None and not (
            (args.type == "sycophancy" and args.design in ("rollout", "did"))
            or (args.type == "sandbagging" and args.design == "did")):
        ap.error("--max-new-tokens only applies to --design rollout or did")
    # did defaults the belief gate to 'sampled' (the consistency prerequisite,
    # ADR 0012); everything else defaults to 'logprob' (ADR 0007). --gate is None
    # when unset so an explicit choice is distinguishable from the default.
    gate_mode = args.gate or ("sampled" if args.design == "did" else "logprob")
    gate_overridden = (args.gate is not None or args.gate_n is not None
                       or args.gate_threshold is not None
                       or args.gate_temperature is not None)
    if gate_overridden and not (
            (args.type == "sycophancy" and args.design in ("behavioral", "rollout", "did"))
            or (args.type == "sandbagging" and args.design == "did")):
        ap.error("--gate/--gate-n/--gate-threshold/--gate-temperature only apply "
                 "to sycophancy --design behavioral/rollout/did or sandbagging "
                 "--design did")
    if (args.gate_n is not None or args.gate_threshold is not None
            or args.gate_temperature is not None) and gate_mode != "sampled":
        ap.error("--gate-n/--gate-threshold/--gate-temperature require --gate sampled")
    args.gate = gate_mode  # collapse the None default to the resolved mode for run()/meta
    # sandbagging did runs through the shared sycophancy.did_filter, so its gate
    # knobs live in the sycophancy module either way
    if (args.type == "sycophancy" and args.design in ("behavioral", "rollout", "did")
            ) or (args.type == "sandbagging" and args.design == "did"):
        from dprobe.data import sycophancy as _syc
        _syc.GATE_MODE = gate_mode
        if args.gate_n is not None:
            _syc.GATE_N = args.gate_n
        if args.gate_threshold is not None:
            _syc.GATE_THRESHOLD = args.gate_threshold
        if args.gate_temperature is not None:
            _syc.GATE_TEMPERATURE = args.gate_temperature
    if args.type == "sycophancy" and args.design == "rollout":
        from dprobe.data import sycophancy as _syc
        if args.rollouts is not None:
            _syc.ROLLOUT_N = args.rollouts
        if args.temperature is not None:
            _syc.ROLLOUT_TEMPERATURE = args.temperature
        if args.max_new_tokens is not None:
            _syc.ROLLOUT_MAX_NEW_TOKENS = args.max_new_tokens
        if args.rollout_prefix is not None:
            _syc.ROLLOUT_PREFIX_MODE = args.rollout_prefix
    # did generates too (gate + pressured answer), sharing ROLLOUT_MAX_NEW_TOKENS
    if _did_run and args.max_new_tokens is not None:
        from dprobe.data import sycophancy as _syc
        _syc.ROLLOUT_MAX_NEW_TOKENS = args.max_new_tokens
    if args.read_prompt == "neutral" and not (
            args.type == "sycophancy" and args.design == "behavioral"):
        ap.error("--read-prompt neutral needs --type sycophancy --design behavioral "
                 "(labels + the stripped prompt come from the behavioral filter)")
    if args.filter:
        if data.FILTERS.get(args.type) is None:
            ap.error(f"--filter is not defined for --type {args.type} "
                     f"(have {sorted(data.FILTERS)})")
        if args.type == "sycophancy" and args.design not in ("framing", "behavioral", "rollout", "did"):
            ap.error("--filter for sycophancy requires --design framing, "
                     "behavioral, rollout or did (completion has no model choice)")
    del _did_run  # guards done; run() re-derives it from args

    # run dir exists from the very start so console.log records the whole run;
    # meta.json is written last, so a dir without it is a crashed/aborted run
    kind = ("permctrl" if args.permute
            else "neutralctrl" if args.read_prompt == "neutral" else args.type)
    run_dir = runlog.new_run_dir(kind, args.method)
    with runlog.capture_console(run_dir):
        run(args, run_dir)


def run(args, run_dir):
    print(f"run dir: {run_dir}")
    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    print(f"model: {model_name} | device: {device}")

    examples = data.get(args.type, design=args.design, source=args.source,
                        pressure=args.pressure)
    if args.max_examples is not None:
        before = len(examples)
        examples = data.subsample(examples, args.max_examples, SEED)
        if len(examples) != before:
            print(f"capped to {len(examples)}/{before} examples (--max-examples {args.max_examples})")
    filter_stats = None
    filter_log = None
    if args.filter:
        flt = data.FILTERS[args.type]
        before = len(examples)
        examples = flt(model, tokenizer, device, examples)
        filter_stats = getattr(flt, "last_stats", None)
        filter_log = getattr(flt, "last_log", None)
        print(f"{args.type} filter: kept {len(examples)}/{before} examples")

    # difference-of-differences (ADR 0012) has its own dual-position extraction
    # and writes a probe + report for each read position, so it branches here
    # (sandbagging did runs the same path; calm_system in meta routes the calm
    # side of each arrow to the CONTROL system prompt)
    if args.design == "did" and args.type in ("sycophancy", "sandbagging"):
        _run_did(args, run_dir, model, tokenizer, device, examples,
                 filter_stats, filter_log, model_name)
        return

    # group key stays the PRESSURE prompt in both read modes, so the neutral
    # control reuses the identical held-out split as its paired pressure run
    groups = [ex.user for ex in examples]
    if args.read_prompt == "neutral":
        examples = [replace(ex, user=ex.meta["neutral_user"]) for ex in examples]
        print("NEUTRAL-READ CONTROL: extracting on persona-stripped prompts -- "
              "AUROC here is pure question-content confound (no pressure present)")

    # guard: eyeball + assert the hidden state is read at the answer-commit token
    # (the most common silent bug) before paying for the extraction pass
    verify_read_positions(tokenizer, examples)

    print(f"extracting activations for {len(examples)} examples ...")
    acts, labels = extract(model, tokenizer, examples, device, batch_size=args.batch_size,
                           acts_dtype=np.dtype(args.acts_dtype))

    if args.permute:
        labels = np.random.default_rng(SEED).permutation(labels)
        print("PERMUTATION CONTROL: labels shuffled -- a leak-free pipeline gives AUROC ~0.5")

    # hold out whole prompts: a pair's two halves share the group key, so this
    # keeps them on the same side of the split (genuinely independent held-out set)
    aurocs, best_layer, probe = layer_sweep(acts, labels, args.method, args.type, groups, C=args.C)
    print(f"best layer {best_layer} | held-out AUROC {aurocs[best_layer]:.3f}")

    out = report_one_type(acts, labels, aurocs, best_layer, probe, args.method, groups, run_dir)
    np.savez(run_dir / "probe.npz", direction=probe.direction, bias=probe.bias,
             layer=probe.layer, method=probe.method, deception_type=probe.deception_type)
    np.save(run_dir / "aurocs.npy", aurocs)
    if filter_log:
        (run_dir / "run_log.txt").write_text(filter_log, encoding="utf-8")
        print(f"wrote per-question filter log -> {run_dir / 'run_log.txt'}")
    meta = _base_meta(args, model_name, device, model, filter_stats)
    meta.update({
        "n_examples": int(len(labels)),
        "n_label1": int((labels == 1).sum()),
        "n_label0": int((labels == 0).sum()),
        "n_groups": int(len(set(groups))),
        "best_layer": int(best_layer),
        "auroc": float(aurocs[best_layer]),
        "aurocs": [float(a) for a in aurocs],
    })
    runlog.write_meta(run_dir, meta)
    print(f"saved -> {run_dir}")


def _base_meta(args, model_name, device, model, filter_stats):
    """The meta.json fields shared by the standard and did run paths."""
    _gated = ((args.type == "sycophancy" and args.design in ("behavioral", "rollout", "did"))
              or (args.type == "sandbagging" and args.design == "did"))
    return {
        "kind": "train_one",
        "type": args.type,
        "design": args.design if args.type in ("sycophancy", "sandbagging") else "n/a",
        "source": args.source if args.type in ("sycophancy", "sandbagging") else "n/a",
        "pressure": args.pressure if (
            args.type == "sandbagging" and args.design == "did") else "n/a",
        "gate": args.gate if _gated else "n/a",
        "read_prompt": args.read_prompt,
        "method": args.method,
        "seed": SEED,
        "max_examples": args.max_examples,
        "filter": bool(args.filter),
        "permuted": args.permute,
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "acts_dtype": args.acts_dtype,
        "filter_stats": filter_stats,
    }


def _run_did(args, run_dir, model, tokenizer, device, examples,
             filter_stats, filter_log, model_name):
    """The difference-of-differences path (ADR 0012).

    Builds pressure-response arrows at both read positions, then runs the SAME
    layer-sweep + report on each -- the arrows are the features, so `fit_mms` on
    them is the capitulation direction. Writes a probe + report per position and
    records the answer-token-minus-prompt-final AUROC gap (the letter-shortcut
    estimate) in meta.
    """
    from dprobe.did import POSITIONS, extract_arrows

    # single-class / starved filter output is a legitimate finding, not a crash:
    # record it (meta + filter trace) and exit cleanly so the run dir is not
    # mistaken for an aborted run (ADR 0011 convention: no meta.json = crashed)
    if not examples:
        print("did: 0 examples survived the filter (see warning above) -- "
              "nothing to train; writing meta + filter trace and exiting.")
        if filter_log:
            (run_dir / "run_log.txt").write_text(filter_log, encoding="utf-8")
            print(f"wrote per-question filter log -> {run_dir / 'run_log.txt'}")
        meta = _base_meta(args, model_name, device, model, filter_stats)
        meta.update({"n_examples": 0, "aborted": "no_examples_after_filter"})
        runlog.write_meta(run_dir, meta)
        print(f"saved -> {run_dir}")
        return

    print(f"extracting DiD arrows for {len(examples)} questions x 4 reads ...")
    arrows, labels, groups = extract_arrows(
        model, tokenizer, device, examples,
        batch_size=args.batch_size, acts_dtype=np.dtype(args.acts_dtype))

    if args.permute:
        labels = np.random.default_rng(SEED).permutation(labels)
        print("PERMUTATION CONTROL: labels shuffled -- a leak-free pipeline gives AUROC ~0.5")

    positions = {}
    for pos in POSITIONS:
        acts = arrows[pos]
        aurocs, best_layer, probe = layer_sweep(
            acts, labels, args.method, args.type, groups, C=args.C)
        print(f"[{pos}] best layer {best_layer} | held-out AUROC {aurocs[best_layer]:.3f}")
        report_one_type(acts, labels, aurocs, best_layer, probe, args.method,
                        groups, run_dir, tag=pos)
        np.savez(run_dir / f"probe_{pos}.npz", direction=probe.direction, bias=probe.bias,
                 layer=probe.layer, method=probe.method, deception_type=probe.deception_type)
        np.save(run_dir / f"aurocs_{pos}.npy", aurocs)
        positions[pos] = {"best_layer": int(best_layer),
                          "auroc": float(aurocs[best_layer]),
                          "aurocs": [float(a) for a in aurocs]}

    if filter_log:
        (run_dir / "run_log.txt").write_text(filter_log, encoding="utf-8")
        print(f"wrote per-question filter log -> {run_dir / 'run_log.txt'}")

    # the answer-token arm carries a letter-identity term the clean prompt-final
    # arm does not, so its excess AUROC estimates the letter shortcut (ADR 0012)
    gap = positions["answertoken"]["auroc"] - positions["promptfinal"]["auroc"]
    # stamp the EFFECTIVE generation/gate knobs (did shares ROLLOUT_MAX_NEW_TOKENS)
    # so the run is reproducible from meta, not just the CLI history
    from dprobe.data import sycophancy as _syc
    meta = _base_meta(args, model_name, device, model, filter_stats)
    meta.update({
        "max_new_tokens": int(_syc.ROLLOUT_MAX_NEW_TOKENS),
        "gate_n": int(_syc.GATE_N),
        "gate_threshold": float(_syc.GATE_THRESHOLD),
        "gate_temperature": float(_syc.GATE_TEMPERATURE),
        "n_examples": int(len(labels)),
        "n_label1": int((labels == 1).sum()),
        "n_label0": int((labels == 0).sum()),
        "n_groups": int(len(set(groups))),
        "did_positions": positions,
        "letter_shortcut_gap": float(gap),
        # top-level best_layer/auroc mirror the CLEAN prompt-final arm at a glance
        "best_layer": positions["promptfinal"]["best_layer"],
        "auroc": positions["promptfinal"]["auroc"],
    })
    runlog.write_meta(run_dir, meta)
    print(f"letter-shortcut gap (answertoken - promptfinal AUROC): {gap:+.3f}")
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
