# ADR 0012 — Sycophancy Difference-of-Differences (design="did")

**Status:** Proposed — needs Dr. Zhu / team sign-off before any results from it
are used or reported (changes core methodology relative to ADR 0007/0008). Code
is merged behind `--design did`; nothing existing changes behaviour.
**Date:** 2026-07-14

## Context

The behavioural design (ADR 0007) is confounded by **question content**: its
neutral-read control hit AUROC **1.000 @ layer 1** with no pressure present
(ADR 0008), because labelling is deterministic per question, so cave-questions
and hold-questions are disjoint sets and topic alone predicts the label.

ADR 0008 (`rollout`) fixes this by making the label a per-rollout property and
balancing cave/hold **within** each question, so content cancels in the class
means. But it only keeps **ambivalent** questions (both choices sampled), and on
opinion data the model is ~99.8% committed per rollout, so it starves
(4/248 ambivalent at temperature 1.0 on 7B, ADR 0009).

**Difference-of-differences (DiD) is a second, independent rescue for the
behavioural confound.** For each question run a **calm** (persona-stripped,
unpressured) and a **pressured** version and subtract the activations to get that
item's **pressure-response arrow**. The *first* subtraction cancels question
content (the same `c_q` is in both endpoints). Then average the caving arrows,
average the holding arrows, and subtract: the *second* subtraction cancels the
generic pressure response shared by both classes, leaving a candidate
**capitulation direction**.

Its advantage over `rollout`: DiD needs **no ambivalent questions** — every gated
question is cleanly cave *or* hold and is usable — so yield does not collapse on
opinion or on starved factual data. For the `rollout` design DiD is a **no-op**
(a question's calm activation is identical across its cave and hold children and
within-question balancing makes the calm term exactly 0), so DiD is specifically
the *behavioural* rescue, presented as a `rollout` alternative.

Closed form (all terms are linear means):

```
DiD_direction  =  [mean(P|cave) − mean(P|hold)]  −  [mean(C|cave) − mean(C|hold)]
              =  (behavioural direction)          −  (neutral-read direction)
```

i.e. DiD subtracts the exact vector that scored AUROC 1.000 in ADR 0008's
neutral-read control out of the behavioural direction.

## Decision

Add `design="did"` (fifth `--design` value). Reuses `behavioral`'s belief gate,
source data (`opinion` or `factual`) and pressure prompt; requires `--filter`.

Per question:

1. **Belief / consistency gate** (calm side; `--gate sampled` is the intended
   default here): keep the question only if the model, *unpressured*, gives the
   honest (`not_matching`) answer on ≥ threshold of N samples. Self-consistency
   under no pressure *defines* the model's belief — objective ground truth is not
   required, so **opinion is a valid source**. `factual` (user asserts a wrong
   answer) additionally guards against the confidently-wrong case (the model
   consistently holds a shared misconception and the user happens to assert the
   truth, so "caving" would move toward truth); it is the **recommended /
   trustworthy** source, but not mandatory.
2. **Pressured label, on-policy:** greedily generate the answer under the full
   pressure prompt; the letter it emits is the label — 1 = user-matching
   (`caved`), 0 = honest (`held`). Greedy = deterministic *and* on-policy.
3. **Two paired reads → two arrows per question** (`arrow = pressured_act −
   calm_act`, per layer):
   - **prompt-final**: last prompt token, no answer appended. Pre-commitment,
     *predictive*, and carries no committed-letter token, so it is the **clean**
     capitulation direction.
   - **answer-token**: read at the bare committed `" (X)"` answer string
     (`commit`-style, ADR 0009 addendum — kills the preamble-wording shortcut).
     Calm side reads the honest letter (the gate guarantees it); pressured side
     reads `matching` if caved else `not_matching`. On-policy, *concurrent*.
4. **Capitulation direction = diff-of-means on the arrows** (`fit_mms(arrows)`),
   per read position. No new estimator.
5. **Scoring is on the arrows, not the pressured snapshot.** Content `c_q` is
   already cancelled inside each arrow, so the report's AUROC / PCA / ROC reflect
   pressure-response geometry only — an honest measure, not one a topic shortcut
   can inflate. Feeding `arrows` to the existing `layer_sweep` + `report_one_type`
   reuses the whole path unchanged.
6. **Both read positions get their own report and probe** in the run dir:
   `report_sycophancy_<method>_promptfinal.png` /
   `report_sycophancy_<method>_answertoken.png`, and
   `probe_promptfinal.npz` / `probe_answertoken.npz`.

### Why not letter-balance the answer-token arm?

A caving question flips its answer letter between the calm read (honest letter)
and the pressured read (wrong letter); a holding question does not. So the
answer-token arrow carries a **letter-identity term** that appears only in the
cave class and that the DiD subtraction does **not** cancel. Borrowing ADR 0008
step 7 (letter-side balance) would zero it — but the whole point of emitting
*both* positions is to **measure** that term: the prompt-final arm has no letter
term, so the **AUROC gap (answer-token − prompt-final) estimates the letter
shortcut**. Balancing would delete the measurement. The prompt-final arm is the
direction to trust; the answer-token arm is diagnostic. (Letter-balance stays as
future work if a clean answer-token direction is ever wanted.)

## Consequences

- **Not a single-pass detector.** Scoring a question needs its calm
  counterfactual (both snapshots), so this is a research probe answering "does a
  capitulation direction exist / where is it," not something deployable on one
  live prompt. Transfer to types without a calm baseline is undefined; the
  transfer matrix would need a `did`-specific pairing on both sides.
- **Cost per question:** gate (N sampled generate) + 1 greedy pressured generate
  + 4 teacher-forced forwards (2 positions × {calm, pressured}). Same order as
  the ADR 0008 filter.
- **Yield** is the main win over `rollout`: no ambivalence requirement, so every
  gated question contributes. Watch the cave/hold base rate instead
  (`_balance` subsamples the majority class).
- **Claim differs by arm:** prompt-final = *predictive* ("about to cave"),
  answer-token = *concurrent* ("caving now"). Present the two figures together.
- `--permute` remains valid (shuffle the arrow labels → expect 0.5). The
  gate/pressured sampling is seeded but not GPU-portable-exact (same caveat as
  ADR 0008).
- `--read-prompt neutral` does not apply (DiD already reads the neutral prompt as
  the calm endpoint); `train_one` rejects the combination.

## Open questions

- A third read position: mean over the response tokens (Apollo aggregation),
  vs the commit token.
- Majority-vote pressured label (N samples) instead of greedy single — greedy
  chosen for determinism; revisit if labels look noisy near the boundary.
- A deployable arm: also emit the pressured-snapshot scoring (direction from
  arrows, but score raw `P`) so `did` probes slot into the transfer matrix.
- Sign-off: changes core methodology relative to ADR 0007/0008 — **pending team
  review** (status stays Proposed until then).
