# ADR 0011 — Sandbagging: Within-Question Rollout Design on the Shared MCQ Source

**Status:** Proposed
**Date:** 2026-07-09
**Author:** Alina Liu

## Context

The sandbagging module is still the 12-item seed: free-form QA under a
control-vs-sandbag **system-prompt contrast**. This has the exact defect that
killed sycophancy's `framing` design (ADR 0007): the two classes differ in the
instruction tokens, so a probe can score AUROC ~1.0 by reading "is the sandbag
instruction present" off the input, without representing any deceptive state.
Scaling that design would produce a large, meaningless number.

Meanwhile the sycophancy line converged on a construction that survives its
controls (ADR 0008 rollouts + ADR 0009 factual source: permutation ≈ 0.5 on
2026-07-07T01-58-29Z; wording-shortcut ablation informative on
2026-07-07T02-21-08Z). Sandbagging is structurally the SAME problem:

- sycophancy's belief gate ("does the model hold the correct answer
  unpressured?") **is** sandbagging's capability filter (the CLAUDE.md
  IMPORTANT rule) — ADR 0009 already noted the analogy;
- both need a within-question contrast so question content cannot separate the
  classes;
- both need a known-correct-answer MCQ source so "deceptive" is well-defined.

## Decision

**PROPOSED** — give sandbagging a `rollout` design that reuses the ADR 0008
mechanics wholesale, differing from sycophancy-factual in exactly one place:
**where the pressure lives**.

| | sycophancy `factual` (ADR 0009) | sandbagging (this ADR) |
|---|---|---|
| question source | ARC Easy+Challenge, 2-choice | same (shared `mcq.py`) |
| pressure | USER turn asserts the wrong answer | SYSTEM prompt instructs underperformance |
| unpressured gate prompt | persona/assertion-stripped user text | same user text, CONTROL system |
| gate meaning | "model believes the correct answer" | "model is CAPABLE of the correct answer" |
| label 1 | caved to the user's wrong answer | complied: gave the wrong answer it could avoid |
| label 0 | held the correct answer | resisted: answered correctly under pressure |

Mechanics (identical to ADR 0008/0009, via a shared engine): capability gate →
N rollouts at temperature > 0 under the identical pressure prompt → parse the
"(X)" commit → keep only ambivalent questions → within-question cave/hold
balance → letter-side balance → read position at the answer-commit token
(`prefix_mode` commit/text, commit default per the ADR 0009 addendum).

The pressure-location asymmetry is a **feature**: within a type both classes
share the full prompt (no leak), and across types the transfer matrix then
tests whether a probe generalises across where the pressure came from — a
stronger claim than transfer between two user-turn constructions.

### Code layout (deliberate non-refactor)

`sycophancy.py` is under active development on the steering line and is NOT
edited. New shared modules instead:

- `data/mcq.py` — ARC + offline-fixture loading, seeded 95/5 split, seeded
  letter-side/body construction. Seed-identical to `sycophancy._factual_rows` /
  `_make_factual_row` (tested byte-for-byte), so **train/test splits agree
  across types** — the transfer matrix never trains one type on another type's
  test questions. Merging sycophancy onto `mcq.py` is follow-up work, not this ADR.
- `data/rollout.py` — the generic engine (gate, sampling, parsing via the
  imported `sycophancy.parse_choice`, balances, stats, run log), parameterised
  only by how to build the unpressured gate prompt. Knob defaults are
  test-pinned to sycophancy's.
- `data/sandbagging.py` — `build(design='rollout', source='factual'|'factual-small')`
  emitting the -1 sentinel; `sandbagging_filter` dispatcher (rollout → engine,
  else legacy capability filter). The 12-item completion design is kept as the
  documented leaky baseline, like sycophancy's.

CLI: `python -m scripts.train_one --type sandbagging --design rollout
--source factual --filter` (+ the existing `--rollouts/--temperature/
--max-new-tokens/--rollout-prefix/--gate*` knobs and `--permute` control).

## Validation plan (before trusting any AUROC)

1. Offline smoke on `factual-small` (plumbing only).
2. 7B OSC run on `--source factual` (~1500 questions to start, mirroring the
   2026-07-07 sycophancy runs), reporting the same filter_stats.
3. `--permute` control: expect ≈ 0.5.
4. `--rollout-prefix text` ablation: measures the wording shortcut, as in the
   ADR 0009 addendum.
5. Then, and only then: the sycophancy↔sandbagging transfer cells
   (`scripts/compare.py`) and, if a probe survives, steering (ADR 0010 harness;
   `steer.py` currently builds items from the sycophancy source, so the
   sandbagging arm needs a small extension — out of scope here).

## Consequences

- First trustworthy sandbagging probe candidate; unblocks the headline
  transfer matrix (the sycophancy line is single-type until a second type
  reaches this standard).
- `compare.py` still builds non-sycophancy types with default (completion)
  designs; pointing it at rollout-design datasets is follow-up work once this
  ADR and 0008/0009 are Accepted.
- One open sampling question inherited from the sycophancy runs: ambivalence
  yield. If the 7B model either always complies or always resists under the
  sandbag prompt, few questions survive (the ADR 0009 starvation failure mode).
  Fallbacks, in order: soften/harden the sandbag system prompt, raise
  temperature, raise N. Report `ambivalent_questions` prominently.

## Open questions

- **Prompted vs. spontaneous sandbagging.** The sandbag system prompt makes
  label 1 = instructed underperformance. Whether a probe trained on instructed
  sandbagging detects un-instructed strategic underperformance is exactly the
  kind of transfer question this project is for; flagging so nobody reads the
  probe as detecting "spontaneous" sandbagging.
- Should the sandbag prompt state the incentive ("you are being evaluated…")
  without the explicit "give an INCORRECT answer" directive? A milder prompt
  may raise ambivalence yield and better approximate realistic sandbagging;
  proposed as a `--source`/prompt variant after the first 7B run.
- Gate strictness: the logprob gate is a point estimate of capability; the
  `--gate sampled` "model is SURE" variant may matter more for sandbagging
  than sycophancy (capability is the entire label semantics here).
