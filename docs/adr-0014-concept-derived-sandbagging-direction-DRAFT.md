# ADR 0014 — Concept-Derived Sandbagging Direction (Persona Vectors)

**Status:** Proposed
**Date:** 2026-07-27
**Supersedes nothing. Reopens:** sandbagging, deferred by `docs/SCOPE.md` on 2026-07-26.

> Drafted in `docs/` because `.claude/` is write-protected in this session, matching the
> precedent of `adr-0013-evaluation-awareness-DRAFT.md`. Move to
> `.claude/docs/adr/0014-concept-derived-sandbagging-direction.md` before committing.

## Context

Every direction this project has produced is fit on behavioural labels. Diff-of-means,
LR, LDA, DiD arrows: all of them take a set of examples labelled 1 = deceptive and
0 = control, and return the axis that separates them. That makes the quality of the
direction a strict function of the quality of the labels.

The 07-16 sandbagging labels are not clean. Cave rate is 52% when the correct answer is
(A) and 28% when it is (B), z = 3.85
(`docs/sandbagging-did-causal-summary-2026-07-16.md`). Part of the "held" class is
failed compliance that landed on the answer key by accident. That bias sits inside the
label, so it sits inside the direction, and detection AUROC measured against the same
labels cannot see it. The causal result from that run (mms @ L20, add pass 0 → 0.72,
ablate 1.0 → 0.835) is the strongest evidence the project has, and it is evidence about
a direction built from contaminated labels.

Correlational controls do not fix this. Permutation tests split leakage. Bag-of-words
tests surface features. Neither addresses a direction whose training signal is partly
the position of the answer key.

Anthropic's persona vectors (arXiv 2507.21509, Aug 2025) take the label out of the loop.
The pipeline consumes a natural-language description of a trait, auto-generates system
prompts that elicit and suppress it, generates responses under both, and takes the
difference in mean activations over the response tokens. No behavioural label enters the
extraction. The paper demonstrates on Qwen2.5-7B-Instruct, which is the model this
project already uses, so the method is known to work at this scale and this checkpoint.

## Decision

**PROPOSED** — build a sandbagging direction from a trait description rather than from
behavioural labels, then test it on two axes the existing harness already measures.

### Extraction pipeline

| Step | What happens | Why it matters |
|---|---|---|
| 1. Trait definition | One paragraph defining sandbagging: deliberately answering worse than capability allows, on a question the model can answer correctly, to appear less capable | The only human input. Frozen in the repo, versioned, quoted in the paper |
| 2. Prompt generation | An external LLM writes 5 trait-eliciting and 5 trait-suppressing system prompts, plus ~40 questions | Automated per the paper. Generated once, reviewed by hand, then **frozen to a JSON fixture** so the direction is reproducible without an API call |
| 3. Response generation | Each of the 10 systems crossed with each question, sampled at T = 1.0 | 400 responses, 200 per pole |
| 4. Judge filter | An LLM judge scores trait expression 0-100 per response. Keep positives scoring high, negatives scoring low, drop the rest | Without this the vector encodes *prompt style*, not the trait. This is the step that makes it a persona vector rather than an instruction-contrast vector |
| 5. Direction | Per layer, mean activation over **response tokens**, then `v_L = mean(pos) − mean(neg)`, unit-normalised | Response-token averaging, not last-token. See "What the codebase is missing" |

Output is a `Probe` with `method="concept"`, so `steer.py` and every `probe.npz`
consumer accept it unchanged.

### Two evaluations

**Causal.** `scripts/steer.py`, unchanged, pointed at the concept `probe.npz`. Add pass
on unpressured items, ablate pass on pressured items, random-direction control on.

**Detection.** Score the held-out DiD sandbagging test split with the concept direction.
This is the load-bearing measurement, because the direction was never fit on those
labels. An honest AUROC here is not circular in the way every previous number in this
project has been.

## Pre-registration

Written before the run, per the rule at the bottom of `SCOPE.md`.

| Measurement | Confirms | Kills |
|---|---|---|
| Add pass, unpressured items correct at baseline | Wrong-rate rises monotonically in alpha and clears 10x the random control at max alpha, parse rate intact | Flat, or within 2x of random. The concept direction is not causal and this design is done |
| Ablate pass, pressured items that cave at baseline | Caving drops by at least 0.10 absolute versus the random control | No drop. Elicitation-only effect. Report as such, do not claim a sandbagging direction |
| Detection AUROC, held-out DiD test split | At least 0.70 | At most 0.60. The concept the description names and the behaviour the labels record are different things. That is a publishable negative and it ends the line |
| **(A)/(B) split of the same AUROC** | Both halves equal within bootstrap CI | A gap of the same sign and size as the 07-16 label gap. Then the bias lives in the model's behaviour, not in the label pipeline, and the framing changes |
| Cosine with the 07-16 mms DiD direction | Reported either way | Nothing kills on this row. High cosine means the behavioural direction was mostly right and the confound rode along. Near zero with good steering means two separate causal axes, which needs explaining before anything gets claimed |

Row 4 is the reason for the whole design. It is also what turns the uncomfortable meeting
disclosure into the motivation for the next experiment.

## Controls

1. **Random unit vector**, same layer. Already in `steer.py` (`--control random`).
2. **Opposing pole.** Steer with `−v`. Caving should fall where `+v` raised it. A
   direction that only works in one sign is a magnitude artifact.
3. **Prompt-style vector.** Build a second direction from the same prompt pairs with the
   judge filter off and the read at the last *prompt* token. If it scores the same as the
   real one, steps 3 through 5 buy nothing and what got measured is instruction style.
   This is the control the `completion` and `framing` designs never had.
4. **KL-matched steering control.** A norm-matched random direction inflates effect size
   (`SCOPE.md` infrastructure table, ~30 lines). The headline here is causal, so this one
   is not optional.
5. **Bootstrap CIs** on every AUROC cell. Bare point estimates are what the 07-26 critique
   flagged.
6. **Second judge.** Rebuild the direction with a different judge model and report the
   cosine between the two. This measures the live risk directly instead of arguing about
   it: if the direction moves when the judge changes, the vector encodes the judge's
   opinion. Primary is `claude-sonnet-5`, second is `gpt-5.4-2026-03-05`. Both differ in
   family from each other and from the subject model, which is what rules out
   `qwen3-32b` as a check on Qwen2.5-7B. Both support `temperature=0` and both are
   pinned rather than preview, so the cached scores stay reproducible for the length of
   the study. Costs one extra extraction and no extra generation.

## Why the 5 past failure modes do not apply

| Past failure | Applies here? |
|---|---|
| Prompt-token leakage (`completion`, `framing`) | **No.** The direction is fit on generated trait prompts and evaluated on ARC-derived items. The two sets share no tokens and no template, so a prompt feature cannot produce evaluation AUROC. It *can* produce a prompt-style vector that steers nothing, which control 3 catches |
| Letter shortcut, answer-token arm | **Structurally impossible.** The extraction set has no forced choice, no letters, no answer key |
| Answer-side bias, (A) vs (B) | **Structurally impossible in the direction.** This is the point of the ADR. It stays live in the evaluation set, which is why it gets measured as a split rather than assumed away |
| Content confounding (`behavioral`) | **Controlled by construction.** The same questions appear on both poles; only the system prompt differs. Topic cannot align with the pole |
| Rollout starvation | **Not applicable.** No rollouts, no per-question class balance requirement |

**The risk that stays live: the judge.** Filtering responses with an LLM judge means the
direction inherits that judge's notion of sandbagging. If the judge is really scoring
"sounds unconfident," the vector is an unconfidence vector wearing a sandbagging label.

Mitigation: hand-score a 50-response sample against the judge and record the agreement
rate in `meta.json`. Under 0.85 agreement, the rubric gets rewritten before the direction
is trusted.

Naming exactly one live risk in advance is the standard `SCOPE.md` set for the scheming
design. This design meets it.

## Two objections to answer before the numbers exist

**"The detection evaluation still uses the contaminated labels, so what did you gain?"**
The gain is one-directional and worth stating precisely. Contaminated labels cannot
inflate this AUROC in the way they inflate a probe fit on them, because the direction
never saw them. But they can *deflate* it. A low AUROC is therefore ambiguous: either
the concept and the behaviour are different things, or the labels are noisy enough to
cap any direction's score. Disambiguation: score the 07-16 mms behavioural probe on the
same split and report both. If the behavioural probe also scores low there, the ceiling
is the labels. This is a cheap extra column and it goes in the same table.

**"The (A)/(B) split will not have the power to show anything."** Probably true at 400
questions. The 07-16 gap was 52% vs 28% on cave *rate*, which is a different and
better-powered quantity than an AUROC difference on two half-sized subsets. So the split
gets reported with bootstrap CIs and read as a direction of evidence, not a test.
The properly powered version is to compare cave rate under `+v` steering across the two
answer-key sides, which reuses the steering run and costs nothing extra. Add that column.

## What the codebase is missing

`activations.extract` reads `h[:, -1, :]`, the last token of the prompt, and nothing
else. Persona vectors average over response tokens. That is the one real gap.

| Piece | Status |
|---|---|
| `extract_response_mean` | **New.** Generate, then mean-pool hidden states over generated tokens only, per layer. ~60 lines |
| Frozen prompt/question fixture | **New.** JSON in `src/dprobe/data/fixtures/` |
| Judge call plus scoring | **New.** ~40 lines |
| `Probe` with `method="concept"` | Reuses `probes.Probe` as-is. It is a direction, a bias, and a layer |
| `scripts/steer.py` | **Unchanged.** Takes any `probe.npz` |
| Detection scoring | Needs `score_fixed_direction` next to `layer_sweep`, since `layer_sweep` fits rather than scores. ~25 lines |
| `runlog` | Unchanged. New `kind="concept"` |

Roughly 150 new lines plus a fixture.

## Consequences

- Sandbagging comes back into scope. `docs/SCOPE.md` is edited deliberately to say so,
  and scheming moves to the deferred list with the same date discipline.
- The 07-16 result still gets disclosed. This ADR does not soften it. It makes the
  disclosure load-bearing: the confound is the stated motivation for the new design.
- A concept direction is not a deployment-grade deception detector. It is built from 10
  system prompts written to name the behaviour. Generalisation beyond that framing is an
  open question, and detection AUROC on held-out behavioural labels is the only evidence
  that speaks to it.
- If the direction steers but does not detect, that result is worth reporting on its own.
  It says the concept the description names is causally real and behaviourally distinct
  from what the label pipeline records, which is a sharper statement than anything the
  cross-type transfer matrix was going to produce.

## References

- Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits in Language
  Models*, arXiv 2507.21509, Aug 2025. Method summary:
  https://www.anthropic.com/research/persona-vectors
- ADR 0010, steering harness, reused unchanged
- ADR 0012, DiD design, source of the evaluation labels
- `docs/sandbagging-did-causal-summary-2026-07-16.md`, the contaminated result
- `docs/critique-2026-07-26.md`, the circularity problem this addresses
