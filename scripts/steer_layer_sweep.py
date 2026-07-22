"""Find the best *steering* layer around the AUROC-peak *detection* layer.

The per-layer AUROC sweep (scripts/train_one.py) tells you where the caving
direction is most *decodable* -- call that L*. But the best layer to *steer* is
usually a few blocks EARLIER: an injection at L* has almost no downstream depth
left to propagate through, whereas one a little earlier is re-read and amplified
by every block above it. Detection and control peak at different depths.

So this script sweeps candidate steering layers in an ASYMMETRIC band around L*
(wider below than above), builds a raw-space diff-in-means caving direction at
each one from the caved/held split, and selects the layer with the strongest
*causal* effect that still leaves the model coherent -- not the best readout.

Pipeline (all on the sycophancy `factual` source, like scripts/steer.py):

  1. TRAIN split -> run the design's filter -> labeled caved(1)/held(0) examples,
     then cache a per-layer feature for each in ONE extraction pass (so every band
     direction is built from the cache -- the practical note in the task). The
     feature depends on --direction:
       did (default)      -- the pressure-response ARROW act(pressured)-act(calm)
                             at the clean prompt-final read (dprobe.did, ADR 0012):
                             the first subtraction cancels question content, so the
                             class-mean difference is the capitulation direction.
                             This is what scripts/steer.py causally validates.
       behavioral/rollout -- the raw activation at the answer-commit token (attach
                             the committed answer as an assistant_prefix; rollout
                             children already carry one). The ADR 0008 behavioural
                             direction (content-confounded), kept as a baseline.
  2. For each band layer L: v_L = unit(mean(caved) - mean(held)) on that feature,
     in RAW residual units (fit_mms, not the z-scored fit_mms_std -- steering
     needs real units). For did, mean-difference ON THE ARROWS is the DiD probe.
  3. TEST split (never used to build a direction) -> causal eval by free
     generation:
       - cave induction: on unpressured prompts the base model HOLDS on, add
         +alpha*v_L; the wrong-answer rate should climb.
       - hold induction: on pressured prompts the base model CAVES on, add
         -alpha*v_L; the caving rate should fall (hold rate climbs).
       v_L is unit-normed; alpha is a fraction of the layer's mean residual norm,
       swept over a small set so no layer is judged at a bad coefficient.
       Answers are classified with the SAME parse_choice judge the split used.
  4. Coherence guardrail: per (layer, alpha) measure perplexity on a held-out
     text sample with the steering hook live, so a layer that only "caves" by
     emitting gibberish is flagged, not selected.
  5. Write a table + plot to the run dir and print the recommended steering layer
     (strongest combined cave+hold effect among coherent layers).

    # offline plumbing check (no network, NOT trustworthy numbers):
    python -m scripts.steer_layer_sweep --best-layer 8 --source factual-small --max-items 8

    # real run on OSC, DiD direction, L* read from a did run's meta.json:
    python -m scripts.steer_layer_sweep --sweep results/runs/<did_run> --model Qwen/Qwen2.5-7B-Instruct --source factual --max-new-tokens 16

Writes its own run dir (kind="steerlayer"); meta.json holds the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dprobe import data, runlog
from dprobe.activations import (
    build_prompt,
    default_batch_size,
    extract,
    load_model,
    verify_read_positions,
)
from dprobe.config import MODEL_NAME, SEED
from dprobe.data.sycophancy import DID_SYSTEM
from dprobe.probes import Probe, fit_mms
from dprobe.tracefmt import render
# steering internals reused so the causal eval is byte-for-byte the same
# generation + judging the rest of the pipeline uses (parse_choice via _classify)
from dprobe.steer import (
    _classify,
    _install,
    _parse_rate,
    _wrong_rate,
    build_items,
    make_add_hook,
)

# Held-out fluent English, unrelated to the steering data: a perturbation that
# only elicits caving by breaking generation blows these up; a real steering
# direction leaves them near baseline. This is the coherence probe (guard 4).
COHERENCE_TEXTS = [
    "The sun rises in the east and sets in the west every single day.",
    "Water is made of hydrogen and oxygen and boils at one hundred degrees "
    "Celsius at sea level.",
    "A short walk after dinner can aid digestion and help you sleep better.",
    "The library opens at nine in the morning and closes at six in the evening.",
    "She packed her bags, locked the front door, and drove to the airport "
    "before dawn.",
    "Regular exercise, a balanced diet, and enough sleep are the foundations "
    "of good health.",
]


def _read_best_layer(spec: str) -> int:
    """L* from a sweep artifact: a run dir, a meta.json, or an aurocs.npy.

    A dir prefers its meta.json's ``best_layer`` (the cross-machine source of
    truth), falling back to argmax of aurocs.npy; a .json reads ``best_layer``;
    a .npy is argmax'd directly.
    """
    p = Path(spec)
    if p.is_dir():
        mj, af = p / "meta.json", p / "aurocs.npy"
        if mj.exists():
            return int(json.loads(mj.read_text())["best_layer"])
        if af.exists():
            return int(np.load(af).argmax())
        raise FileNotFoundError(f"{p} has neither meta.json nor aurocs.npy")
    if p.suffix == ".json":
        return int(json.loads(p.read_text())["best_layer"])
    if p.suffix == ".npy":
        return int(np.load(p).argmax())
    raise ValueError(f"--sweep {spec!r}: expected a run dir, .json, or .npy")


def _answer_token_examples(examples):
    """Make every labeled example read at the answer-commit token.

    Rollout children already carry an ``assistant_prefix`` (the sampled commit
    token); behavioural ones do not, so attach the answer the model actually
    committed under pressure -- ``matching`` if it caved (label 1), else
    ``not_matching``. The extraction then reads the "(X)" itself (the DiD
    answer-token read), matching the split the AUROC sweep was built on.
    """
    out = []
    for ex in examples:
        if "assistant_prefix" in ex.meta:
            out.append(ex)
            continue
        prefix = ex.meta["matching"] if ex.label == 1 else ex.meta["not_matching"]
        out.append(replace(ex, meta={**ex.meta, "assistant_prefix": prefix}))
    return out


def _layer_scales(model, tokenizer, items, device, batch_size):
    """Mean last-token residual L2 norm at EVERY layer over ``items``' prompts.

    One batched forward with output_hidden_states caches all layers at once, so a
    per-layer alpha (a fraction of the layer's typical activation norm) costs no
    extra passes. Returns an array indexed by hidden-state layer (0 = embeddings).
    """
    prompts = [build_prompt(tokenizer, it.example) for it in items]
    order = sorted(range(len(prompts)), key=lambda i: len(tokenizer.encode(prompts[i])))
    sums = None
    n = 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [prompts[i] for i in idx]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # [n_layers+1, batch] last-token norms
        norms = torch.stack([h[:, -1, :].float().norm(dim=-1)
                             for h in out.hidden_states], dim=0)
        s = norms.sum(dim=1).cpu().numpy()
        sums = s if sums is None else sums + s
        n += len(idx)
    return sums / max(n, 1)


def _perplexity(model, tokenizer, texts, device, layer=None, vec=None, coeff=0.0):
    """Geometric-mean perplexity over ``texts``, optionally with a steering hook.

    With ``vec``/``coeff`` set, ``coeff * vec`` is added at ``layer`` for the whole
    forward, so the returned PPL is the model's fluency UNDER that intervention --
    the coherence signal. Each text is scored unpadded (labels = its own ids), so
    left-padding never leaks into the loss.
    """
    handle = (_install(model, layer, make_add_hook(vec, coeff))
              if vec is not None and coeff else None)
    try:
        nlls = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**ids, labels=ids["input_ids"])
            nlls.append(float(out.loss))
    finally:
        if handle is not None:
            handle.remove()
    return float(np.exp(np.mean(nlls)))


def _select_targets(items, base_counts, want):
    """Split baseline generations into the subset each pass acts on.

    ``want="hold"`` keeps items the base model answers CORRECTLY unpressured (the
    cave-induction targets -- a flip is only meaningful from a held baseline);
    ``want="cave"`` keeps items it CAVES on under pressure (the hold-induction
    targets). Items with no parseable baseline answer are dropped either way.
    Returns (kept_items, their_baseline_counts) so the caller can read the
    baseline rate straight off the selected subset.
    """
    keep, counts = [], []
    for it, c in zip(items, base_counts):
        if c["wrong"] + c["correct"] == 0:
            continue
        if (want == "hold" and c["correct"] > c["wrong"]) or (
                want == "cave" and c["wrong"] > c["correct"]):
            keep.append(it)
            counts.append(c)
    return keep, counts


def _letter_of(answer):
    """The bare choice letter in a matching/not_matching string like ' (B)' -> 'B'."""
    m = re.search(r"\(([A-Za-z])\)", answer)
    return m.group(1).upper() if m else None


def _split_by_letter(items, counts, caving):
    """Break a cave/hold pass down by the ASSERTED-WRONG letter (it.matching) -- the
    B-pusher probe.

    A cave is a flip from the model's correct answer to the asserted-wrong one, so a
    cave on a wrong='(B)' item is an A->B flip and on a wrong='(A)' item a B->A flip.
    A genuine capitulation direction caves at ~equal rates for both letters; a
    token-'B' pusher caves almost only when the wrong answer is B (A->B) and, on the
    hold pass, rescues almost only the B-caves. Returns {letter: {"rate", "n"}} where
    rate is the cave rate (caving=True) or hold rate (caving=False) within the group
    of items whose asserted-wrong answer is that letter, and n is that group size."""
    groups = {}
    for it, c in zip(items, counts):
        let = _letter_of(it.matching)
        if let is not None:
            groups.setdefault(let, []).append(c)
    out = {}
    for let, cs in groups.items():
        wr = _wrong_rate(cs)
        out[let] = {"rate": (wr if caving else 1.0 - wr), "n": len(cs)}
    return out


def _letter_fields(cave_split, hold_split):
    """Flatten the two per-letter dicts into the flat row fields (nan/0 if absent).
    cave_B is the A->B flip rate, cave_A the B->A rate; hold_A/hold_B are the rescue
    rates among the A-caves / B-caves respectively."""
    def g(split, let, key, default):
        return split.get(let, {}).get(key, default)
    return {
        "cave_A": g(cave_split, "A", "rate", float("nan")),
        "cave_B": g(cave_split, "B", "rate", float("nan")),
        "n_cave_A": g(cave_split, "A", "n", 0),
        "n_cave_B": g(cave_split, "B", "n", 0),
        "hold_A": g(hold_split, "A", "rate", float("nan")),
        "hold_B": g(hold_split, "B", "rate", float("nan")),
        "n_hold_A": g(hold_split, "A", "n", 0),
        "n_hold_B": g(hold_split, "B", "n", 0),
    }


def _split_str(s):
    """Compact ASCII A/B breakdown for a per-layer summary dict (cave A->B is the
    wrong='(B)' group, B->A the wrong='(A)' group; hold rescue split the same way)."""
    def fmt(rate, n):
        return (f"{rate:.3f}(n={n})" if not (isinstance(rate, float) and np.isnan(rate))
                else f"n/a(n={n})")
    return (f"cave A->B {fmt(s['cave_B'], s['n_cave_B'])} | "
            f"B->A {fmt(s['cave_A'], s['n_cave_A'])}   "
            f"hold rescue A-cave {fmt(s['hold_A'], s['n_hold_A'])} | "
            f"B-cave {fmt(s['hold_B'], s['n_hold_B'])}")


def _pusher_verdict(s, ratio=3.0):
    """One-line read on the cave pass: balanced (genuine capitulation) vs a
    letter-token pusher. Compares the A->B and B->A cave rates."""
    ab, ba = s["cave_B"], s["cave_A"]  # A->B rate, B->A rate
    if any(isinstance(x, float) and np.isnan(x) for x in (ab, ba)) or (
            s["n_cave_A"] == 0 or s["n_cave_B"] == 0):
        return "A/B verdict: one letter group is empty -- cannot judge pusher bias"
    hi, lo = max(ab, ba), min(ab, ba)
    if hi == 0:
        return "A/B verdict: no caves in either letter group at this alpha"
    # lo == 0 with hi > 0 is the MOST skewed case (a pure token-pusher), so it must
    # trip the flag -- guarding on lo > 0 would wrongly wave the perfect pusher through
    if lo == 0 or hi / lo >= ratio:
        skew = "A->B" if ab > ba else "B->A"
        letter = "B-pusher" if ab > ba else "A-pusher"
        factor = "inf" if lo == 0 else f"{hi / lo:.1f}x"
        return (f"A/B verdict: cave skews {skew} ({hi:.3f} vs {lo:.3f}, {factor}) "
                f"-- possible residual {letter}, not pure capitulation")
    return (f"A/B verdict: cave A->B {ab:.3f} ~ B->A {ba:.3f} (<{ratio:g}x) "
            "-- consistent with genuine capitulation, not a letter-pusher")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--best-layer", type=int, default=None,
                     help="L*, the AUROC-peak DETECTION layer, given directly.")
    src.add_argument("--sweep", default=None,
                     help="read L* from a sweep artifact instead: a train_one run "
                          "dir (uses meta.json best_layer), a meta.json, or an "
                          "aurocs.npy (argmax).")
    ap.add_argument("--n-before", type=int, default=8,
                    help="layers to sweep BELOW L* (default 8). Wider than "
                         "--n-after because the best steering layer sits earlier "
                         "than the best readout layer. Clamped to layer >= 1.")
    ap.add_argument("--n-after", type=int, default=3,
                    help="layers to sweep ABOVE L* (default 3). Clamped to the "
                         "model's last layer.")
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id; MUST be the model L* was measured "
                         "on (a direction and a depth are model-specific). Default: "
                         "config.MODEL_NAME.")
    ap.add_argument("--direction", default="did",
                    choices=["did", "behavioral", "rollout"],
                    help="which construction the per-layer steering direction comes "
                         "from: 'did' (default) = diff-of-means on pressure-response "
                         "ARROWS at the clean prompt-final read (ADR 0012, the "
                         "capitulation direction steer.py validates); 'behavioral' = "
                         "diff-in-means on raw answer-token activations (ADR 0008, "
                         "content-confounded baseline); 'rollout' = same as "
                         "behavioral but per-sampled-answer labels. did is "
                         "forced-choice by construction (DID_SYSTEM), so its eval "
                         "runs forced-choice automatically -- --forced-choice is for "
                         "the behavioral/rollout directions.")
    ap.add_argument("--source", default="factual", choices=["factual", "factual-small"],
                    help="question source: 'factual' = ARC (needs network); "
                         "'factual-small' = offline repo fixture (plumbing only).")
    ap.add_argument("--gate", default="logprob", choices=["logprob", "sampled"],
                    help="belief gate for the caved/held labeling (ADR 0007/0012): "
                         "'logprob' (default here -- cheap, no generation in the "
                         "gate) or 'sampled' (train_one's did default, the 'model is "
                         "SURE' consistency gate; costs a generate call per "
                         "question). Only affects which questions build the "
                         "direction, not the causal eval.")
    ap.add_argument("--gate-n", type=int, default=None,
                    help="--gate sampled only: samples per question (default 20). "
                         "Use 10 to match the did_qwen.sbatch labeling exactly.")
    ap.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.5,2,2.5,3,4",
        help="comma-separated add strengths, each a MULTIPLE of the layer's mean "
             "residual norm -- same scaling and default distribution as "
             "scripts/steer.py's add pass, so the two are directly comparable. "
             "The coherent regime is alpha <~1.5 (higher breaks the (A)/(B) "
             "format); the dense sub-1.25 sampling is where the signal lives.")
    ap.add_argument("--coherence-threshold", type=float, default=0.5,
                    help="minimum coherence ratio (baseline_ppl / steered_ppl) for "
                         "an (layer, alpha) to count as coherent. 0.5 = steered "
                         "perplexity may at most double before the point is flagged "
                         "as gibberish and excluded from the recommendation.")
    ap.add_argument("--parse-floor", type=float, default=0.5,
                    help="minimum fraction of steered completions that must still "
                         "yield a parseable choice for an (layer, alpha) to count "
                         "-- a second, generation-side degeneration guard.")
    ap.add_argument("--forced-choice", action="store_true",
                    help="constrain the model to a bare '(A)'/'(B)' answer "
                         "(DID_SYSTEM) for BOTH direction-building and eval, so the "
                         "caving measurement is not corrupted by free-form "
                         "truncation/rebuttals (ADR 0012). Pair with a small "
                         "--max-new-tokens (e.g. 16).")
    ap.add_argument("--samples", type=int, default=1,
                    help="completions per item: 1 = greedy (default), >1 samples "
                         "at --temperature for a smoother rate.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature when --samples > 1.")
    ap.add_argument("--max-new-tokens", type=int, default=24,
                    help="generation cap per completion (must reach the '(X)').")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap the number of TRAIN questions used to build the "
                         "directions (random whole-prompt subsample) for speed.")
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap the number of TEST questions used for the causal "
                         "eval (random whole-prompt subsample) for speed.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="generation/extraction batch size; default auto-picks "
                         "from GPU VRAM.")
    args = ap.parse_args()

    run_dir = runlog.new_run_dir("steerlayer", args.direction)
    with runlog.capture_console(run_dir):
        run(args, run_dir)


def run(args, run_dir):
    print(f"run dir: {run_dir}")
    best_layer = args.best_layer if args.best_layer is not None else _read_best_layer(args.sweep)
    print(f"L* (detection peak) = {best_layer}"
          + ("" if args.best_layer is not None else f"  (from {args.sweep})"))

    model_name = args.model or MODEL_NAME
    model, tokenizer, device = load_model(model_name)
    batch_size = args.batch_size or default_batch_size(device)
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"model: {model_name} | device: {device} | batch_size: {batch_size}")

    # ---- 1. TRAIN split -> labeled caved/held -> cached per-layer feature ----
    # did carries its forced-choice system by construction; behavioral/rollout can
    # be forced with --forced-choice so the eval regime matches the direction's.
    from dprobe.data import sycophancy as _syc
    _syc.GATE_MODE = args.gate  # belief-gate mode for the labeling filter
    if args.gate_n is not None:
        _syc.GATE_N = args.gate_n
    train_examples = data.get("sycophancy", design=args.direction, source=args.source,
                              split="train")
    if args.forced_choice and args.direction != "did":
        train_examples = [replace(e, system=DID_SYSTEM) for e in train_examples]
    if args.max_train is not None:
        train_examples = data.subsample(train_examples, args.max_train, SEED)
    print(f"labeling {len(train_examples)} train questions ({args.direction} filter) ...")
    labeled = data.FILTERS["sycophancy"](model, tokenizer, device, train_examples)
    lab0 = np.array([e.label for e in labeled])
    if set(lab0.tolist()) != {0, 1}:
        raise ValueError(
            f"need both caved(1) and held(0) after filtering, got labels "
            f"{sorted(set(lab0.tolist()))} over {len(labeled)} examples -- widen "
            "--max-train / --source or loosen the gate")
    print(f"  {len(labeled)} labeled ({int((lab0==1).sum())} caved / "
          f"{int((lab0==0).sum())} held)")

    # cache a per-layer feature per example in ONE pass; the feature is the DiD
    # arrow (default) or the raw answer-token activation (behavioral/rollout)
    if args.direction == "did":
        from dprobe.did import extract_arrows
        print(f"extracting DiD prompt-final arrows for {len(labeled)} questions ...")
        arrows, labels, _ = extract_arrows(model, tokenizer, device, labeled,
                                           batch_size=batch_size, positions=("promptfinal",))
        feats = arrows["promptfinal"]
    else:
        at_examples = _answer_token_examples(labeled)
        verify_read_positions(tokenizer, at_examples)
        print(f"extracting all-layer activations for {len(at_examples)} examples ...")
        feats, labels = extract(model, tokenizer, at_examples, device, batch_size=batch_size)
    n_layers = feats.shape[1] - 1  # feature has n_layers+1 columns (0 = embeddings)

    # ---- 2. asymmetric band; one raw diff-in-means direction per band layer ----
    lo = max(1, best_layer - args.n_before)
    hi = min(n_layers, best_layer + args.n_after)
    band = list(range(lo, hi + 1))
    print(f"steering band: layers {lo}..{hi} "
          f"(L*-{best_layer-lo}, L*+{hi-best_layer}) of 1..{n_layers}")
    directions = {L: torch.from_numpy(
        fit_mms(feats[:, L, :], labels, L, "sycophancy").direction.astype(np.float32))
        for L in band}

    # ---- 3a. TEST split; ONE baseline generation selects each pass's targets ----
    eval_examples = data.get("sycophancy", design=args.direction, source=args.source,
                             split="test")
    if args.forced_choice and args.direction != "did":
        eval_examples = [replace(e, system=DID_SYSTEM) for e in eval_examples]
    if args.max_items is not None:
        eval_examples = data.subsample(eval_examples, args.max_items, SEED)
    gen = dict(batch_size=batch_size, max_new_tokens=args.max_new_tokens,
               samples=args.samples, temperature=args.temperature)

    unpressured = build_items(eval_examples, pressured=False)
    pressured = build_items(eval_examples, pressured=True)
    print(f"baseline generation on {len(eval_examples)} held-out test questions ...")
    base_unpressured = _classify(model, tokenizer, unpressured, device, **gen)
    base_pressured = _classify(model, tokenizer, pressured, device, **gen)
    cave_targets, cave_base = _select_targets(unpressured, base_unpressured, want="hold")
    hold_targets, hold_base = _select_targets(pressured, base_pressured, want="cave")
    base_cave_rate = _wrong_rate(cave_base) if cave_targets else float("nan")
    base_hold_rate = (1.0 - _wrong_rate(hold_base)) if hold_targets else float("nan")
    print(f"  cave-induction targets (base holds): {len(cave_targets)} | "
          f"hold-induction targets (base caves): {len(hold_targets)}")
    if not cave_targets and not hold_targets:
        raise ValueError("no usable causal-eval targets: the base model neither "
                         "holds on any unpressured item nor caves on any pressured "
                         "item. Widen --max-items or check the source/model.")

    # per-layer residual scale (one cached forward per target subset)
    cave_scale = (_layer_scales(model, tokenizer, cave_targets, device, batch_size)
                  if cave_targets else None)
    hold_scale = (_layer_scales(model, tokenizer, hold_targets, device, batch_size)
                  if hold_targets else None)
    base_ppl = _perplexity(model, tokenizer, COHERENCE_TEXTS, device)
    print(f"baseline perplexity (coherence reference): {base_ppl:.2f}")

    # ---- 3b/4. sweep every (layer, alpha): cave, hold, coherence ----
    rows = []  # one dict per (layer, alpha)
    for L in band:
        v = directions[L]
        print(f"layer {L}:")
        for a in alphas:
            cave_rate = cave_parse = float("nan")
            coh_cave = 1.0
            cave_split, hold_split = {}, {}
            if cave_targets:
                coeff = a * float(cave_scale[L])
                handle = _install(model, L, make_add_hook(v, coeff))
                try:
                    c = _classify(model, tokenizer, cave_targets, device, **gen)
                finally:
                    handle.remove()
                cave_rate, cave_parse = _wrong_rate(c), _parse_rate(c)
                cave_split = _split_by_letter(cave_targets, c, caving=True)
                coh_cave = base_ppl / _perplexity(model, tokenizer, COHERENCE_TEXTS,
                                                  device, L, v, coeff)
            hold_rate = hold_parse = float("nan")
            coh_hold = 1.0
            if hold_targets:
                coeff = a * float(hold_scale[L])
                handle = _install(model, L, make_add_hook(v, -coeff))  # -alpha*v
                try:
                    c = _classify(model, tokenizer, hold_targets, device, **gen)
                finally:
                    handle.remove()
                hold_rate = 1.0 - _wrong_rate(c)  # holding = not caving
                hold_parse = _parse_rate(c)
                hold_split = _split_by_letter(hold_targets, c, caving=False)
                coh_hold = base_ppl / _perplexity(model, tokenizer, COHERENCE_TEXTS,
                                                  device, L, v, -coeff)
            coherence = min(coh_cave, coh_hold)
            parse = np.nanmin([cave_parse, hold_parse])
            coherent = coherence >= args.coherence_threshold and parse >= args.parse_floor
            cave_gain = cave_rate - base_cave_rate
            hold_gain = hold_rate - base_hold_rate
            combined = float(np.nansum([cave_gain, hold_gain])) if not (
                np.isnan(cave_gain) and np.isnan(hold_gain)) else float("nan")
            rows.append({"layer": L, "alpha": a, "cave_rate": cave_rate,
                         "hold_rate": hold_rate, "cave_gain": cave_gain,
                         "hold_gain": hold_gain, "combined": combined,
                         "coherence": coherence, "cave_parse": cave_parse,
                         "hold_parse": hold_parse, "coherent": bool(coherent),
                         **_letter_fields(cave_split, hold_split)})
            print(f"    alpha={a:>5.2f}  cave={cave_rate:.3f} hold={hold_rate:.3f} "
                  f"combined={combined:+.3f}  coherence={coherence:.2f} "
                  f"parse={parse:.2f}  {'OK' if coherent else 'FLAG'}")

    # ---- per-layer pick: strongest combined effect at a COHERENT alpha ----
    summary = _per_layer_summary(band, rows)
    coherent_layers = [s for s in summary if s["coherent"] and not np.isnan(s["combined"])]
    recommended = (max(coherent_layers, key=lambda s: s["combined"])["layer"]
                   if coherent_layers else None)

    print("\nper-layer best coherent alpha:")
    for s in summary:
        tag = "" if s["coherent"] else "  (no coherent alpha)"
        star = "  <-- recommended" if s["layer"] == recommended else ""
        print(f"  layer {s['layer']:>3}  alpha={s['best_alpha']:>5.2f}  "
              f"cave={s['cave_rate']:.3f} hold={s['hold_rate']:.3f} "
              f"combined={s['combined']:+.3f} coherence={s['coherence']:.2f}"
              f"{tag}{star}")
        print(f"           {_split_str(s)}")
    if recommended is not None:
        print(f"\nRECOMMENDED steering layer: {recommended} "
              f"(L* was {best_layer}; {best_layer - recommended:+d} below)")
        rs = next(s for s in summary if s["layer"] == recommended)
        print(f"  A/B balance @ L{recommended}: {_split_str(rs)}")
        print("  " + _pusher_verdict(rs))
    else:
        print("\nNo layer produced a coherent causal effect at any swept alpha -- "
              "widen --alphas, relax --coherence-threshold, or inspect run_log.txt.")

    # ---- 5. artifacts: table, plot, trace, meta ----
    _write_csv(run_dir / "steer_layer_sweep.csv", rows)
    plot_path = _plot(run_dir, band, summary, best_layer, recommended,
                      base_cave_rate, base_hold_rate, args)
    (run_dir / "run_log.txt").write_text(
        _render_log(args, best_layer, recommended, base_cave_rate, base_hold_rate,
                    base_ppl, summary, rows), encoding="utf-8")
    runlog.write_meta(run_dir, {
        "kind": "steer_layer_sweep",
        "type": "sycophancy",
        "direction": args.direction,
        "source": args.source,
        "gate": args.gate,
        "gate_n": args.gate_n if args.gate == "sampled" else None,
        "forced_choice_effective": bool(args.forced_choice or args.direction == "did"),
        "model": model_name,
        "device": device,
        "model_dtype": str(model.dtype),
        "best_layer_detection": int(best_layer),
        "recommended_steering_layer": recommended,
        "band": [int(lo), int(hi)],
        "n_before": args.n_before,
        "n_after": args.n_after,
        "alphas": alphas,
        "forced_choice": args.forced_choice,
        "coherence_threshold": args.coherence_threshold,
        "parse_floor": args.parse_floor,
        "max_new_tokens": args.max_new_tokens,
        "samples": args.samples,
        "temperature": args.temperature,
        "seed": SEED,
        "n_train_labeled": int(len(labeled)),
        "n_cave_targets": len(cave_targets),
        "n_hold_targets": len(hold_targets),
        "base_cave_rate": None if np.isnan(base_cave_rate) else float(base_cave_rate),
        "base_hold_rate": None if np.isnan(base_hold_rate) else float(base_hold_rate),
        "base_perplexity": base_ppl,
        "per_layer": summary,
        "rows": rows,
    })
    print(f"saved -> {run_dir}\n  table: {run_dir / 'steer_layer_sweep.csv'}\n  "
          f"plot:  {plot_path}")


def _per_layer_summary(band, rows):
    """Per layer: the coherent alpha with the strongest combined effect (or, if no
    alpha stayed coherent, the most-coherent one, flagged) -- the plotted point."""
    out = []
    for L in band:
        rl = [r for r in rows if r["layer"] == L]
        coherent = [r for r in rl if r["coherent"] and not np.isnan(r["combined"])]
        if coherent:
            best = max(coherent, key=lambda r: r["combined"])
        else:
            best = max(rl, key=lambda r: (r["coherence"]
                                          if not np.isnan(r["coherence"]) else -1))
        out.append({"layer": L, "best_alpha": best["alpha"],
                    "cave_rate": best["cave_rate"], "hold_rate": best["hold_rate"],
                    "combined": best["combined"], "coherence": best["coherence"],
                    "coherent": best["coherent"],
                    **{k: best[k] for k in ("cave_A", "cave_B", "n_cave_A",
                                            "n_cave_B", "hold_A", "hold_B",
                                            "n_hold_A", "n_hold_B")}})
    return out


def _write_csv(path, rows):
    cols = ["layer", "alpha", "cave_rate", "hold_rate", "cave_gain", "hold_gain",
            "combined", "coherence", "cave_parse", "hold_parse", "coherent",
            "cave_A", "cave_B", "n_cave_A", "n_cave_B",
            "hold_A", "hold_B", "n_hold_A", "n_hold_B"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(int(r[c]))
            if c != "coherent" else str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(run_dir, band, summary, best_layer, recommended, base_cave, base_hold, args):
    """Layer on x; cave- and hold-induction rate (at each layer's best coherent
    alpha) on the left axis, coherence overlaid on the right. L* and the
    recommended layer are marked; incoherent layers get hollow markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [s["layer"] for s in summary]
    cave = [s["cave_rate"] for s in summary]
    hold = [s["hold_rate"] for s in summary]
    coh = [s["coherence"] for s in summary]
    solid = [s["coherent"] for s in summary]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, cave, "-o", color="#c4452f", label="cave induction (+αv)")
    ax.plot(xs, hold, "-o", color="#2a7fb8", label="hold induction (−αv)")
    # hollow markers where no alpha stayed coherent
    for x, yc, yh, ok in zip(xs, cave, hold, solid):
        if not ok:
            ax.scatter([x, x], [yc, yh], facecolors="white",
                       edgecolors=["#c4452f", "#2a7fb8"], zorder=5, s=55)
    if not np.isnan(base_cave):
        ax.axhline(base_cave, ls=":", color="#c4452f", lw=1, alpha=0.6)
    if not np.isnan(base_hold):
        ax.axhline(base_hold, ls=":", color="#2a7fb8", lw=1, alpha=0.6)
    ax.axvline(best_layer, ls="--", color="gray", lw=1.2, label=f"L* (detect) = {best_layer}")
    if recommended is not None:
        ax.axvline(recommended, ls="-", color="#2ca02c", lw=1.6,
                   label=f"recommended = {recommended}")
    ax.set(xlabel="steering layer", ylabel="induced rate (at best coherent α)",
           ylim=(-0.03, 1.03), title="Causal steering-layer sweep around L*")

    ax2 = ax.twinx()
    ax2.plot(xs, coh, "-s", color="#7f7f7f", alpha=0.7, ms=4, label="coherence")
    ax2.axhline(args.coherence_threshold, ls=":", color="#7f7f7f", lw=1)
    ax2.set_ylabel("coherence (base_ppl / steered_ppl)")
    ax2.set_ylim(0, 1.05)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = run_dir / "steer_layer_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _render_log(args, best_layer, recommended, base_cave, base_hold, base_ppl,
                summary, rows):
    """Full per-(layer, alpha) trace for run_log.txt: every point's cave/hold
    rate, coherence, parse rates and coherent verdict, plus the per-layer pick."""
    stats = {
        "L*_detection": best_layer,
        "recommended_steering_layer": recommended,
        "direction": args.direction, "source": args.source,
        "forced_choice": args.forced_choice,
        "alphas": args.alphas,
        "coherence_threshold": args.coherence_threshold,
        "parse_floor": args.parse_floor,
        "base_cave_rate": round(float(base_cave), 4) if not np.isnan(base_cave) else "n/a",
        "base_hold_rate": round(float(base_hold), 4) if not np.isnan(base_hold) else "n/a",
        "base_perplexity": round(base_ppl, 3),
    }
    pick = ["legend: per layer, the coherent alpha with the strongest combined "
            "cave+hold effect (or most-coherent if none coherent, flagged)"]
    for s in summary:
        flag = "" if s["coherent"] else "  [NO COHERENT ALPHA]"
        star = "  <-- RECOMMENDED" if s["layer"] == recommended else ""
        pick.append(f"layer {s['layer']:>3}  best_alpha={s['best_alpha']:>5.2f}  "
                    f"cave={s['cave_rate']:.3f} hold={s['hold_rate']:.3f} "
                    f"combined={s['combined']:+.3f} coherence={s['coherence']:.2f}"
                    f"{flag}{star}")
        pick.append(f"          {_split_str(s)}")
    grid = ["legend: cave=+αv wrong-rate, hold=1-(−αv wrong-rate), combined=gain "
            "over baseline, coherence=base_ppl/steered_ppl, OK/FLAG per guardrail"]
    for r in rows:
        grid.append(
            f"layer {r['layer']:>3} alpha={r['alpha']:>5.2f}  "
            f"cave={r['cave_rate']:.3f} hold={r['hold_rate']:.3f} "
            f"combined={r['combined']:+.3f}  coherence={r['coherence']:.2f} "
            f"parse(c/h)={r['cave_parse']:.2f}/{r['hold_parse']:.2f}  "
            f"{'OK' if r['coherent'] else 'FLAG'}")
    return render("STEERING-LAYER SWEEP RUN LOG", stats,
                  [("PER-LAYER PICK", pick), (f"FULL GRID ({len(rows)} points)", grid)])


if __name__ == "__main__":
    main()
