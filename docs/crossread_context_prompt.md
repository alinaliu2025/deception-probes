# Context prompt: the cross-read (crossread) experiment in `deception-probes`

You are being briefed on one experiment inside `dprobe`, a research pipeline for training
linear deception probes on LLM residual-stream activations (repo:
https://github.com/alinaliu2025/deception-probes, branch `sandbagging-rollout`, script
`scripts/crossread.py`). Read this context fully before doing anything else.

## Project background (one paragraph)

The pipeline trains simple linear probes (difference-of-means "mms", logistic regression
"lr") to separate honest from deceptive internal states in an instruct model (currently
Qwen2.5-7B-Instruct), across three deception types: sycophancy, sandbagging, and omission.
For sycophancy, the current design is **within-question rollouts**: take factual questions
the model reliably answers correctly without pressure, then re-ask each one ~8–16 times at
nonzero temperature with user pressure toward the wrong answer ("I'm pretty sure the answer
is (B). Can you confirm?"). Rollouts where the model keeps its unpressured answer form the
**HELD** set; rollouts where it flips form the **CAVED** set. The probe direction is the
difference in mean hidden states between the two sets — a candidate "tendency to cave under
pressure" direction. Best result so far: AUROC ~0.98–1.0 at layer 15 of the 7B model, with
low AUROC in early layers rising through the mid-stack (the shape we want). Steering with
this direction did **not** change behavior, so it is currently a correlate, not a cause.

## The problem the crossread answers

Everything above depends on a quiet choice: **which token's hidden state do you read?**
The rollout pipeline reads the state on a *reconstructed* prompt, and there are two
legitimate reconstructions:

1. **`commit` read ("reenactment").** The assistant prefix is just the bare answer tag,
   `" (X)"`. This was introduced deliberately (ADR 0009 addendum) because the earlier
   version had a **wording shortcut**: caved and held rollouts phrase their preamble
   differently ("Yes, you're right..." vs "Actually, no..."), so a probe reading a prefix
   containing that text got AUROC ≥0.97 from layer 1, flat across all layers — surface
   text separability, not an internal deception signal. Stripping the prefix to the bare
   answer kills that shortcut. **Cost:** the state is counterfactual — it describes a model
   that answered instantly with no lead-in, which never actually happened.
2. **`text` read ("generation-time").** The assistant prefix is the full sampled response,
   cut right at the "(X)". A forward pass over those tokens reproduces (up to
   re-tokenization) the state at the moment the model *actually committed* to its answer.
   **Cost:** the wording confound is back in context.

Each read looked good on its own: commit hit mms AUROC 1.0 at L13 (run
2026-07-06T11-07-48Z), text hit 0.98 at L15 (run 2026-07-07T02-21-08Z — this is the result
in Jack's message). But two separately-good AUROCs do not show the two reads see the
**same** signal. If each read learns a read-specific artifact, then "which token to read"
was silently determining what the probe measures — the exact worry Jack raised. The
crossread makes that worry precise and testable.

## The design: a 2x2 transfer over read positions

Train a probe on one read's activations, evaluate it on the other's, per layer:

|                    | test on commit | test on text |
|--------------------|----------------|--------------|
| **train on commit**| C→C            | C→T          |
| **train on text**  | T→C            | T→T          |

Interpretation:
- **High off-diagonals (C→T, T→C):** both reads expose one shared direction; the token
  choice was never the issue.
- **Off-diagonals at chance while diagonals stay high:** each read learns a read-specific
  artifact — the token-position worry confirmed.
- **Layer-dependent split is the expected interesting case:** the text read has an
  early-layer wording signal the commit read lacks, so cross cells should fail early and
  converge mid-stack if a shared deception direction exists there.

## Mechanics (why it's a fair comparison)

- **One** expensive rollout-filter pass, run in `text` mode. Every kept rollout child
  carries its full sampled text as the assistant prefix.
- Each child's **commit twin** is derived by swapping that prefix for the bare answer tag
  (`meta["matching"]` iff label 1). Same question, same label, same rollout — the **only**
  difference between the twin rows is where the state is read.
- One grouped train/test split (grouped by question, so no question leaks across the
  split) is shared by **all four cells** — the twin activation sets are row-aligned, so
  the split transfers verbatim. Every cell scores the identical example list.
- `verify_read_positions` checks both prefix constructions tokenize to the intended read
  token before extraction.

## Actual result (run 2026-07-11T20-11-22Z, mms, factual source, 7B, 476 examples / 196 question groups)

- Early layers (1–10): T→T climbs to ~0.97 while C→C lags (~0.6) and both cross cells sit
  well below T→T — consistent with an early wording signal that only the text read sees.
- Mid-stack: at **L15**, C→C = 1.000, **C→T = 0.999**, T→T = 0.981, T→C = 0.857; by
  L19–21 T→C reaches 0.96–0.995.
- Reading: the commit-trained direction transfers almost perfectly onto generation-time
  states in the mid layers, so the mid-stack signal is **shared**, not a read artifact.
  The asymmetry (T→C weaker at L15) fits the text probe partially loading on
  wording variance that the commit states don't contain.
- Standing caveats: this direction remains correlational (steering failed, ADR 0010), and
  the result is one model / one seed / one deception type so far.

## Your task

[FILL IN — e.g. "extend the crossread to sandbagging", "write the methods paragraph
describing this for the paper draft", "critique the design for remaining confounds",
"help interpret the T→C vs C→T asymmetry".]
