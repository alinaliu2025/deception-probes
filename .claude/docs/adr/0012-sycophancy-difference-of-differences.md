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
3. **Up to two paired reads → one arrow per read position per question**
   (`arrow = pressured_act − calm_act`, per layer). Prompt-final is extracted by
   default; answer-token is opt-in via `--both-positions` (see the addendum
   below):
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
6. **Each extracted read position gets its own report and probe** in the run dir:
   `report_sycophancy_<method>_promptfinal.png` /
   `report_sycophancy_<method>_answertoken.png`, and
   `probe_promptfinal.npz` / `probe_answertoken.npz`. `meta.json` records which
   positions were read in `did_read_positions`.

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

### Dual letter orderings + asserted-letter balance (amendment, 2026-07-21)

The claim above that "the prompt-final arm has no letter term" was **wrong**, and
it explains why the first large Qwen2.5-7B `did` run produced a **B-pusher**
rather than a capitulation direction. The prompt-final arm carries a *different*
letter term than the answer-token arm: not the *committed* letter (there is no
committed answer at prompt-final) but the **asserted** letter — the pressure
prompt literally contains *"I'm pretty sure the answer is (B)…"*, and that token
is inside every `pressured` endpoint, hence inside every arrow. It only becomes
the *direction* if the asserted letter correlates with the label, and on Qwen it
does strongly (it caves far more often when the cave answer is (B); the rollout
runs measured cave-letter mass like `{A: 100, B: 506}`). The old
`_balance` was label-only, so `mean(arrow|cave) − mean(arrow|hold)` ≈
`mean("(B) asserted") − mean("(A) asserted")`. `--permute` does **not** catch
this: shuffling labels destroys the letter correlation too.

Two-part fix:

1. **Both letter orderings per question** (`_factual_orderings` in
   `data/sycophancy.py`): every factual question is emitted twice — correct-as-(A)
   and correct-as-(B) — with the user asserting the WRONG letter in each (so
   label 1 = caved is unchanged). Both rows share a `group` id (the source
   question index) that the grouped split keys on, so a question's two orderings
   never straddle the train/test boundary. This balances the asserted letter in
   the **dataset**.
2. **Asserted-letter balance after labeling** (`_balance_by_letter`, replacing the
   label-only `_balance` in `did_filter`): dual ordering balances the dataset, but
   the caved/held label is assigned by the model, so a letter-biased model still
   yields a letter-skewed caved class. Equalising the four
   `(label, asserted-letter)` cells decorrelates the asserted letter from the
   label, so the term cancels in the class means. `meta.json`'s `filter_stats`
   gains `label_letter_cells` (the pre-balance per-cell counts). Falls back to
   plain label balance if a cell is empty (single class, or a letter that never
   caves).

This is orthogonal to the answer-token argument above: that balance was declined
for the *committed*-letter term (to keep the diagnostic gap); this one equalises
the *asserted*-letter term, which lives in **both** arms and is a pure confound in
neither's favour. The answer-token gap diagnostic is therefore preserved.

**Scope:** dual ordering is a property of the `factual`/`factual-small` source, so
`rollout` and `behavioral` runs on that source now also see both orderings —
harmless (rollout already letter-balances the committed letter; behavioral gains
letter balance in its questions) and it doubles the raw question count before
gating. `opinion` source is unaffected (no `group` key → grouping falls back to
the pressure prompt, balance falls back to `_balance`).

### Forced-choice answer format (added after the first 7B factual run)

The first real run (Qwen2.5-7B, `--source factual`, 400 items) exposed a labeling
bug that made the result uninterpretable: **41 of 61 "caved" labels were actually
holds.** Two free-form failure modes did it:

1. **Negation-led rebuttals.** Under pressure the 7B often *rejects* the user in
   prose — *"The correct answer is not (B) ancient fossils… it is (A)."* The old
   `parse_choice` took the **first** `(X)`, which is the rejected letter (the
   user's asserted answer), and scored the refutation as a capitulation.
2. **Reasoning past the token budget.** Chatty answers ran past `max_new_tokens`
   before committing, inflating the unparsed/dropped counts (both gate and
   pressure stages).

This corrupted both arms: prompt-final collapsed to chance (0.545 — the caved
class was 67% mislabeled) and answer-token hit a **circular** 1.000 (its prefix is
built from the *label*, so a wrong label reads a fabricated committed token).

**Decision: `did` probes the FORCED-CHOICE regime.** `DID_SYSTEM` appends
*"Answer with only \"(A)\" or \"(B)\" and nothing else"*, shared by the calm and
pressured sides (so it cancels in the arrow) and by the belief gate (the gate now
reads `ex.system`, not a hardcoded neutral one). This makes the committed answer
unambiguous, kills the truncation loss (the 7B smoke went to **0 unparsed**), and
cleans the gate. `parse_choice` is *also* made negation-aware (skip a `(X)` with a
preceding `not`/`neither`/`nor`/… cue; `infer_opposite` picks the other option
when a two-choice answer only rejects one) as a **fallback** for format
disobedience — and because the same bug was quietly undercounting the gate and
`rollout_filter`.

**The trade-off, explicitly:** forced-choice removes chain-of-thought, so it
probes a *different behavioural regime* than free-form answering — the cave rate
is not comparable across the two, and a weak model with no room to reason may cave
on nearly everything (the 0.5B offline fixture now does, going single-class). We
chose forced-choice because Phase 1 is a controlled *detection* study and
forced-choice `(A)/(B)` is the standard sycophancy-eval setup (Perez et al.
model-written-evals; Anthropic sycophancy work). Free-form sycophancy (parser fix
only, verbose answers kept) is the alternative and is left as future work; a run
must not mix the two.

### Prompt-final by default, answer-token opt-in (addendum)

Originally every `did` run extracted both arms unconditionally. That makes the
*diagnostic* arm cost the same as the *clean* one on every run: four reads per
question instead of two, i.e. **2x the extraction pass and 2x the activation
RAM** — the dominant cost of a large run.

The letter-shortcut gap is a property of the design, not of each run: once
measured on a given model + source, re-measuring it on every subsequent run buys
nothing. So the default is now the clean **prompt-final arm alone**, and
`--both-positions` opts into the answer-token arm and the gap. Nothing about the
arrows, labels, gate, or the direction estimator changed — a default-position run
produces the *identical* `probe_promptfinal.npz` it would have before.

Consequences: `meta.json` gains `did_read_positions`, and
`letter_shortcut_gap` is `null` on runs that read one arm (absent measurement, not
a zero shortcut). Runs are still comparable on the prompt-final arm; the gap is
only comparable among `--both-positions` runs.

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
- **Free-form vs forced-choice** (resolved to forced-choice, see above) — but
  whether the transfer-matrix study should ALSO carry a free-form `did` arm to
  compare regimes is open.
- **Gate `min-parseable` guard.** Forced-choice largely fixes truncation, but the
  gate can still pass on a thin denominator (a `1/1`-parseable "sure"); consider
  requiring a floor of parseable samples before a pass counts.
- A deployable arm: also emit the pressured-snapshot scoring (direction from
  arrows, but score raw `P`) so `did` probes slot into the transfer matrix.
- Sign-off: changes core methodology relative to ADR 0007/0008 — **pending team
  review** (status stays Proposed until then).
