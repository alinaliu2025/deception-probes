# ADR 0008 — Sycophancy Within-Question Rollouts (design="rollout")

**Status:** Proposed — needs Dr. Zhu / team sign-off before any results from it
are used or reported. Code is merged behind `--design rollout`; nothing
existing changes behaviour.
**Date:** 2026-07-02

## Context

The behavioural design (ADR 0007) is confounded by **question content**. The
neutral-read control built for it (`--read-prompt neutral`: keep the
filter-assigned labels, extract on the persona-stripped prompt where no
pressure exists) came back maximally bad on Qwen2.5-7B-Instruct:

- Pressure run `2026-07-02T17-06-14Z` (n=418): AUROC **0.985** @ layer 23,
  already 0.946 at layer 1.
- Neutral control `2026-07-02T17-34-53Z` (n=104): AUROC **1.000 @ layer 1** —
  with no pressure present and nothing sycophantic possible.

Question content alone predicts the labels, so the pressure run's AUROC is
explained by *which question this is*, not by a deception state. Root cause:
labelling is deterministic (teacher-forced log-prob comparison), so the label
is a fixed function of the question — cave-questions and hold-questions are
disjoint sets, and at 7B's 0.79 cave base rate the 21% held questions are a
topically unusual minority. This is exactly ADR 0007's "residual caveat",
now confirmed severe, and its named follow-up ("within-question temperature
sampling") is what this ADR does.

## Decision

Add `design="rollout"` (fourth `--design` value). Same source data, belief
gate and pressure prompt as `behavioral`; the label becomes a **per-rollout**
property instead of a per-question one, so the same question can appear in
both classes and content cannot separate them.

Per question:

1. **Belief gate** (unchanged from ADR 0007): elicit the model's stance on the
   persona-stripped question by `seq_logprob`; keep only if it defaults to the
   honest answer.
2. **Sample N rollouts** (default 8) at temperature (default 1.0) under the
   identical pressure prompt, short `max_new_tokens` (the prompts end with
   `Answer:`; the model answers with `(A)`/`(B)` immediately).
3. **Parse** each rollout: first `(X)` in the generated text, matched against
   the two pre-written answers. Unparseable rollouts are dropped and counted.
4. Keep only **ambivalent** questions — both choices appear among the parsed
   rollouts. Questions where the model always caves or always holds
   contribute one class only and are dropped (that determinism is what
   created the ADR 0007 confound).
5. **Label per rollout**: 1 = user-matching choice (caved), 0 = honest choice
   (held). CLAUDE.md invariant preserved.
6. **Balance within question**: keep k = min(#cave, #hold) rollouts of each
   class (seeded, exact-duplicate texts deduped first). Every kept question
   contributes both classes equally → content is uninformative by
   construction.
7. **Letter-side balance across questions**: a question's cave-letter is
   fixed (the `matching` answer's letter), so cave/hold examples end in
   opposite letters within a question. The raw files are 50/50 A/B
   (verified 2026-07-02) but the gate + ambivalence filters can skew that,
   which would hand the probe a trivial "which letter" shortcut. Equalise
   the kept pair mass between cave-letter-A and cave-letter-B questions so
   letter identity carries zero label signal globally.
8. **Read position: the choice-commit token.** Each rollout's text is
   truncated at the end of its first `(X)` and attached as an assistant
   prefix; extraction reads the last token as usual. The read is therefore
   *during the model's own answer, at the moment of commitment* — the
   answer-token position, consistent with the Phase 1 read-position
   convention.

This is the Apollo-probes construction (Goldowsky-Dill et al., 2502.03407,
**preprint**) adapted to single-turn MCQ: label by what the model actually
does per sampled response, read inside that response.

### Why not read pre-commit (prompt-final or letter-minus-one)?

All rollouts of a question share the prompt, and activations are a
deterministic function of the input text. At prompt-final the two classes'
activations are *identical* within a question — nothing to separate. At the
token just before the letter the prefix is usually identical too (` (`),
same problem. The first position where cave and hold rollouts can differ is
the commit token itself. Post-commit is a *concurrent* probe ("is the model
caving right now"), a weaker claim than predictive but an honest one.

### What the probe can and can't shortcut on

- Letter identity: killed globally by step 7. Within a question the letter
  *is* the behaviour; mapping letter→label requires computing "does this
  choice match the view the user pushed", which is the construct.
- Question content: killed by step 6 (both classes per question, equal
  counts).
- Split leakage: group key stays the pressure-prompt string, so all rollouts
  of a question land on one side of the grouped split, and `--max-examples`
  subsampling keeps rollout groups whole.

## Consequences

- Claim changes from predictive ("about to cave", prompt-final) to
  concurrent ("caving now", answer-token). Present this to the team as a
  scope change, not just an implementation detail.
- Cost per question: 2 teacher-forced forwards (gate) + one batched
  `generate` call (N return sequences, ≤8 new tokens). Same order as the
  ADR 0007 filter.
- Yield risk: n depends on the **ambivalence rate** at the chosen
  temperature. At 7B's 0.79 deterministic base rate many questions may be
  all-cave across 8 samples. Check the printed `ambivalent_questions` stat
  before scaling; raise `--rollouts` or temperature if yield collapses.
- Sampling is stochastic: rollout sets are seeded (`torch.manual_seed`) but
  reproducibility across GPU/driver versions is not guaranteed, unlike the
  deterministic ADR 0007 filter. `meta.json` records n_rollouts and
  temperature.
- `--read-prompt neutral` does not apply (labels are per-rollout, not
  per-question); `train_one` rejects the combination. `--permute` remains
  valid. The natural new control is a **within-question label shuffle**
  (expect 0.5) — future work.
- The tiny-seed-dataset warning stands: do not trust AUROC until this runs
  at scale on a GPU.

## Open questions

- Read position variants: commit token (this ADR) vs mean over response
  tokens (Apollo's aggregation) — worth an ablation if the commit-token
  signal is weak.
- Temperature: 1.0 maximises ambivalence yield but is off greedy policy;
  0.7–0.8 is closer to deployment sampling. Flagged `--temperature`.
- Belief gate still uses a single deterministic elicitation; a sampled
  belief distribution would be more consistent with the rollout philosophy.
- Sign-off: this changes core methodology relative to ADR 0007 —
  **pending team review** (status stays Proposed until then).
