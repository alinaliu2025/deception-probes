# ADR 0006 — Sycophancy Instruction-Contrast Design (no-leak)

**Status:** Accepted
**Date:** 2026-06-29

## Context

The original sycophancy dataset (ADR 0001, `design="completion"`) builds each pair
from one model-written-evals row by **teacher-forcing a pre-written answer string**
as the assistant turn:

- label 1 = `question` + `answer_matching_behavior` (e.g. `" (B)"`, sycophantic)
- label 0 = `question` + `answer_not_matching_behavior` (e.g. `" (A)"`, honest)

Probes read the **last token** (`extract` → `h[:, -1, :]`). With a completion set,
`build_prompt` appends the assistant turn (`add_generation_prompt=False`), so the
read position sits **after the pasted answer**. Causal self-attention folds the
answer (A vs B) into the final-token activation, so the probe just reads *which
answer was pasted*. This is **construct label leakage**, not a code bug.

Symptoms: on the scaled 7B run
(`2026-06-29T00-25-17Z_sycophancy_lr_3da7279_p0329`) AUROC = 0.9987 and the
per-layer curve **plateaus ~0.998 from layer ~16** — the signature of a leak (a
real signal ramps and peaks mid-late, it doesn't slam into a ceiling and sit).

The decisive point: in this design **the model never chooses anything**. Both
answers are dataset fields, teacher-forced. The probe therefore cannot be measuring
the model's deception — the model made no decision. Existing guards don't catch it:
`GroupShuffleSplit` and `--permute` only detect *train/test split* leakage; this
leak is in the *construct*.

The other two types don't have this problem: `sandbagging` and `omission` build
pairs that differ **only in the system prompt**, carry **no completion** (read at
prompt-final, before any answer), and — for sandbagging — gate on a
`capability_filter` that runs the model and keeps only items where the labeled
behaviour is real.

## Decision

**ACCEPTED** — add a second sycophancy construction, `design="framing"` (the
**instruction contrast**), selectable via a new `--design {completion,framing}`
flag. The default stays `completion`, so the existing path and ADR-0001 baseline
are untouched.

In `framing`, each question yields two examples that differ **only in the system
prompt**, with **no completion**:

| label | system prompt | role |
|---|---|---|
| **1** | `NEUTRAL_SYSTEM` ("You are a helpful assistant.") | sycophancy-prone condition (deceptive) |
| **0** | `HONEST_SYSTEM` (explicit honesty priming) | control |

This keeps the CLAUDE.md invariant (label 1 = deceptive condition, 0 = control) and
makes sycophancy consistent with sandbagging/omission instead of special-cased.

Because both classes still differ in *instruction text*, a `behavior_filter`
(analogue of `capability_filter`) gates the set: keep a question only if, scoring
the two pre-written MCQ answers by sequence log-prob, the model prefers the
**sycophantic** answer under `NEUTRAL_SYSTEM` **and** the **honest** answer under
`HONEST_SYSTEM` — i.e. the framing actually flips the choice. Supporting pieces:

- `activations.seq_logprob(model, tokenizer, prompt, continuation, device)` —
  teacher-forced total log-prob of a continuation, for the MCQ choice decision.
- `data.FILTERS` registry + generic `--filter` dispatch in `train_one`
  (replaces the sandbagging-only special-case).
- `data.get(type, design=...)` forwards the design to the sycophancy builder.

**Relation to ADR 0001:** this partially revisits 0001's response-final decision —
*for sycophancy only*. The leak is exactly the response-final read combined with a
pasted answer. `framing` reverts sycophancy to a **prompt-final** read (no
completion). ADR 0001's response-final position remains correct for genuinely
generated completions; it is the *pasted, pre-labeled* answer that leaks, not the
read position alone.

## Consequences

- New `--design` flag (sycophancy-only; ignored with a warning elsewhere) and a
  `design` field in `meta.json`, so framing vs completion runs are self-identifying.
- `--filter` now dispatches through `data.FILTERS` and works for sycophancy
  (`framing` only — `completion` has no model choice to filter and errors out).
- `framing` build and `behavior_filter` need the two candidate answers; they ride
  in `Example.meta` (`matching` / `not_matching`).
- `behavior_filter` is ~4 model forwards per question (2 answers × 2 systems);
  it's a one-time pass like `capability_filter`. Could be batched later if slow at
  scale.
- Expected on a trustworthy run: a **lower, believable** AUROC with a smooth
  mid-late ramp — NOT a layer-16 plateau at 0.99.

## Open questions

- **Residual instruction-token confound.** Even after filtering, the two classes
  differ in system-prompt text and the read position is after it, so the probe
  could read "was an honesty instruction present" rather than "is the model about
  to be sycophantic." The filter ensures the behaviour is *real*, but doesn't
  remove this confound. This is shared with sandbagging/omission and is the main
  reason the stronger **behavioural-split** alternative (Option A: one neutral
  prompt per question, label by the model's *actual* MCQ choice, identical prompts
  across classes) is noted for follow-up. Not built here.
- **Which framing is "deceptive."** We use neutral (latent sycophancy) vs
  honesty-primed. An explicit *sycophancy-priming* label-1 prompt (symmetric to
  sandbagging's explicit sandbag prompt) is an untested alternative.
- **Sum vs mean log-prob** in `seq_logprob`: the two answers are near-identical in
  token structure (" (A)" vs " (B)"), so summed log-prob is a fair comparison here;
  revisit if answer strings ever vary in length.
