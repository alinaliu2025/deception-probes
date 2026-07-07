# ADR 0010 — Steering: Causal Validation of a Probe Direction

**Status:** Proposed
**Date:** 2026-07-06

## Context

The rollout+factual sycophancy pipeline (ADR 0008/0009) produces a diff-in-means
direction whose held-out detection AUROC hit **1.0** at layers 13–15 on the 7B run
(`2026-07-06T11-07-48Z`, prefix_mode=commit). AUROC that high is a red flag, not a
victory: the read position is the choice-commit token ` (X)`, so a direction that
merely encodes **"the committed letter equals the letter the user asserted"** — a
lexical match/no-match relation present whenever output echoes input — would score
1.0 without representing any deceptive *state*. The letter balance de-correlates
the raw letter from the label globally, but not this relational feature.

Detection is correlational and cannot distinguish a **cause** of caving from a
**spectator** feature that co-occurs with it. The correlational controls we have
(`--permute`, `--rollout-prefix text`) test split leakage and the wording shortcut
respectively; neither rules out the letter-match relation, because that relation
lives at the commit token in every one of those variants.

The one test that separates cause from correlation is **intervention**: push the
direction into the residual stream and measure whether behaviour moves.

## Decision

**PROPOSED** — add a steering harness that treats the probe direction as a causal
lever, in two passes on the sycophancy `factual` source (known-correct MCQ, user
asserts the wrong answer):

| Pass | Prompt | Intervention | Signal that the direction is causal |
|---|---|---|---|
| **add** (elicit) | UNPRESSURED question | add `alpha * v` at the probe layer | wrong-answer rate rises monotonically with `alpha` on items correct at baseline |
| **ablate** (suppress) | PRESSURED prompt | project `v` out (`h -= (h·v)v`) | caving rate drops vs. baseline on items that cave at baseline |

A lexical match/no-match artifact cannot pass either: adding it does not coherently
push the model to *wrong* answers, and removing it does not release a pressured
model from caving. A genuine caving direction should do both.

Design choices:

- **Layer indexing.** The probe reads `hidden_states[L]` = the output of decoder
  block `L` (index 0 is embeddings), so we intervene on `model.model.layers[L-1]`
  via a forward hook. Layer 0 is rejected (no block produces it).
- **Alpha scaling.** By default `alpha` is a multiple of the layer's mean
  last-token residual norm, so the same `--alphas` range is meaningful on 0.5B and
  7B. `--raw` switches to absolute `alpha * unit_v`.
- **Baseline gating by generation, not the filter.** The add pass keeps only items
  the model answers correctly unsteered (a flip is undefined otherwise); the ablate
  pass keeps only items it caves on. `steer.py` selects these itself, so no label
  filter is run.
- **Provenance.** Steering loads a saved `probe.npz`, which is gitignored and lives
  only on the machine that produced it — a run must `scp` the `.npz` back from OSC.
  A direction is model-specific, so `--model` must match the training model.

## Planned artifacts

1. `src/dprobe/steer.py` — `load_probe`, forward-hook factories (`make_add_hook`,
   `make_ablate_hook`), `add_sweep`, `ablate_pass`.
2. `scripts/steer.py` — CLI: `--probe`, `--model`, `--source {factual,factual-small}`,
   `--split`, `--mode {add,ablate,both}`, `--alphas`, `--raw`, `--samples`. Writes a
   `kind="steer"` run dir with the curves in `meta.json` + `add_curve.npy`.
3. `data.get(..., split=...)` — forward the split through the registry (steering
   defaults to the `test` split so it intervenes on fresh items).

## Consequences

- This is the plan's Step 5, previously deferred to a stub. It is a real
  methodology addition on top of the still-Proposed ADR 0008/0009, so it stays
  **Proposed** until the rollout/factual base is signed off (Zhu, Zhu, Liu).
- Runs on a real GPU with the 7B model (the direction it validates was trained
  there); `factual-small` gives an offline plumbing check only.
- The causal claim is only as clean as the baseline subset selection — a small
  ambivalent pool (the ADR 0009 motivation) also shrinks the steering sets, so
  report `n_items` alongside every curve.

## Open questions

- Whether a monotone add-curve on a *content-confounded* direction could still
  arise from steering the model toward the asserted-letter token generally; a
  matched control that steers a random unit vector of equal norm is the natural
  baseline (not yet built).
- Best-layer choice for steering vs. detection: the highest-AUROC layer is not
  necessarily the most causally potent one. Sweep is future work.
