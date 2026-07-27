# SCOPE — concept-derived deception directions, 2026-07-27 to 2026-09-06

This is the contract. It is the only document that says what I am working on. If
something is not in here, I am not doing it, and the way to change that is to edit
this file deliberately, not to drift.

Branch: `concept-vectors`, cut from `sandbagging-did`. Model: Qwen2.5-7B-Instruct.
Duration: 6 weeks.

---

## Amendment log

Every scope change gets a line here, with a date and a reason. If this list gets long,
that is the diagnosis.

- **2026-07-26 → 2026-07-27, one day.** Original scope was in-context scheming on
  Qwen3-8B. Replaced with concept-derived directions on sandbagging (ADR 0014). Reason:
  the scheming design pays 6 weeks and a model download to answer a question about
  Apollo's CoT-monitoring lower bound, and it does not touch the problem that actually
  broke the last 4 weeks. The problem is that every direction in this project is fit on
  behavioural labels, and the 07-16 sandbagging labels are contaminated by which side
  the answer key sits on. Persona vectors build the direction from a trait description
  instead, so the contamination cannot enter. That is a fix for the failure mode I
  actually have.

  The honest counter-argument, recorded because it might be right: this is the 6th
  design in 5 weeks and the pattern of abandoning a design one day after writing it
  down is itself the problem. The reason to accept the change anyway is that the new
  design reuses the harness that already works (`steer.py`, ADR 0010) instead of
  building a new one, so the cost of being wrong is about 150 lines rather than 6 weeks.

---

## The one thing

**Does a direction built from a natural-language description of sandbagging, with no
behavioural label anywhere in its construction, steer the behaviour and detect it?**

Both halves matter and they can come apart.

Steering says the concept is causally real inside the model. Detection says the concept
the description names is the same thing the behavioural labels record. A direction that
steers but does not detect is a result on its own, and a sharper one than the cross-type
transfer matrix was ever going to produce.

Full design and pre-registration:
`docs/adr-0014-concept-derived-sandbagging-direction-DRAFT.md`.

---

## Why this design cannot repeat the last 5 weeks

Stated explicitly, because I have built 5 datasets that leaked and I need to know in
advance why this one is different.

| Past failure | Does it apply here? |
|---|---|
| Prompt-token leakage (`completion`, `framing`) | **No.** The direction is fit on generated trait prompts and scored on ARC-derived items. No shared tokens, no shared template. A prompt feature cannot produce evaluation AUROC |
| Letter shortcut (`did`, answer-token arm) | **Structurally impossible.** The extraction set has no forced choice and no letters |
| Answer-side bias, (A) vs (B) | **Structurally impossible in the direction.** This is the point. It stays live in the evaluation set, so it gets measured as a split rather than assumed away |
| Content confounding (`behavioral`) | **Controlled by construction.** The same questions appear on both poles. Only the system prompt differs, so topic cannot align with the pole |
| Rollout starvation (`rollout`) | **Not applicable.** No rollouts |

**The one risk that stays live: the judge.** Filtering responses by trait expression
with an LLM judge means the direction inherits that judge's notion of sandbagging. If
the judge is really scoring "sounds unconfident," what I get is an unconfidence vector
with a sandbagging label on it. Mitigation is in ADR 0014: hand-score 50 responses,
report the agreement rate, rewrite the rubric under 0.85.

One risk remains live and I know which one it is. That's the difference from last time.

---

## Week 1: the run that can kill this

**Question:** does the concept direction move behaviour at all?

**Run:** extract the direction from the frozen fixture, then `scripts/steer.py` on the
held-out DiD sandbagging split, both passes, random control on. This is 1 extraction run
plus 1 steering run on hardware I already have, on a model I already have. No download.

**Pre-registered decisions, in the ADR 0014 table.** Short version:

- **Add pass flat, or within 2x random** → the concept direction is not causal. Report
  the negative, stop. I find this out in week 1.
- **Add pass causal, ablate pass null** → elicitation-only. Real but weaker. Report it
  as elicitation, do not call it a sandbagging direction.
- **Both causal, detection AUROC ≥ 0.70** → the main line. Go to the (A)/(B) split,
  which is the headline.
- **Both causal, detection AUROC ≤ 0.60** → the concept and the behaviour are different
  things. That is publishable and it is the more interesting outcome.

**The (A)/(B) split is the second, independent gate.** Concept AUROC should be equal on
the correct-answer-(A) half and the correct-answer-(B) half. If it is not, and the gap
has the same sign as the 07-16 label gap, then the bias is in the model's behaviour and
not in my label pipeline. That changes what I have to say about the 07-16 result, and I
need to notice it in week 1 rather than in the writeup.

---

## Weeks 2-6

2. Controls before any number is trusted: opposing pole, prompt-style vector
   (`--no-judge --read prompt-final`), KL-matched steering control, judge hand-agreement.
   A direction that fails these gets fixed or dropped, not reported.

3. Bootstrap CIs on every AUROC cell. Layer sensitivity: report the curve, not the max.

4. Dose-response. Vary the number of system prompts per pole and the judge threshold,
   confirm the steering effect moves monotonically. Cheap, and it separates a real
   direction from a lucky fixture.

5. Port the same pipeline to sycophancy from its own trait description. One extra
   extraction run. If both concept directions work, the cross-type cosine is a
   transfer result with no label contamination on either side, which is the 3x3 matrix
   idea done properly at 2x2.

6. Write up.

---

## Infrastructure: about 110 lines, once, and 5 of the critique's problems go away

Unchanged from the 07-26 version. These are one-time fixes to shared code. Do them in
week 1 alongside the extraction.

| Fix | Where | Lines | Retires |
|---|---|---|---|
| Three-way split (train / layer-select / test) | `evaluate.py: layer_sweep` | ~15 | Selecting the best layer on the test set and reporting that max |
| Bootstrap CIs on AUROC | `evaluate.py` | ~20 | Every bare point estimate in the project |
| Persist activations to the run dir | `runlog.py` | ~40 | Re-running GPU jobs to answer questions that are linear algebra on data I already computed |
| `prereg` field, required by `write_meta` | `runlog.py` | ~5 | Runs with no stopping rule |
| KL-matched steering control | `steer.py` | ~30 | Norm-matched controls that inflate the reported effect size |

The three-way split matters less here, because the layer is selected on the extraction
set and evaluated on a different dataset entirely. Do it anyway, it is 15 lines and the
sycophancy port in week 5 needs it.

---

## Explicitly out of scope until 2026-09-06

Not cancelled. Deferred, with a date. If I want to move something up here, I edit this
file and add a line to the amendment log.

- **In-context scheming** and the Qwen3-8B download. The design in
  `docs/pilot-questions-answered.md` and
  `docs/protocol-walkthrough-one-scenario.md` stays as written. `scheming.py` gets
  committed and tagged, not deleted
- Omission
- Eval-awareness (`evalaware.py`), and note the detection result is already published
  (Nguyen et al. 2507.01786), so this was never as novel as it felt
- The full 3x3 cross-type transfer matrix. The 2x2 concept version in week 5 is in
- The model ladder (0.5B → 14B)
- Any new deception type built from behavioural labels

---

## Parking lot

New ideas go here with a date. They do not go into the branch. Reviewed on 2026-09-06.

- (2026-07-26, **promoted to the main line 2026-07-27, ADR 0014**) Concept-derived
  directions à la Anthropic's persona vectors.
- (2026-07-26, **folded into ADR 0014 controls**) Opposing-pole control instead of only
  a random direction.
- (2026-07-26, **folded into week 4**) Dose-response check.
- (2026-07-26) Audit Apollo's public RoleplayDeception set with the leak protocol. The
  honest and deceptive completions differ in topic, not just honesty, so it probably
  fails bag-of-words. Worth a paper section if the main line leaves room.
- (2026-07-26) Nonlinear probe on saved activations, to check what linearity costs.
- (2026-07-27) Persona-vector data flagging, section 3 of the paper: score training or
  eval items by projection onto the concept direction to find the ones that induce the
  trait. Would give a principled way to filter the ARC pool instead of the capability
  filter.
- (2026-07-27) Preventative steering during finetuning. Out of reach on this compute
  budget, but it is the part of the paper with the clearest safety story.

---

## The one meeting

I skip approval and just try things, which is why 3 ADRs have been stuck at Proposed for
weeks while results from them got reported anyway. The fix is not asking 12 times. It's
asking once, about scope. One agenda, one meeting, Thursday 2026-07-30:

1. **Scope.** Concept-derived directions on sandbagging, 6 weeks, per ADR 0014. Say out
   loud that the scheming scope was written on Monday and replaced on Tuesday, and say
   why. Get a yes or a counter-proposal. If the counter-proposal is scheming, take it,
   the design is written and nothing is lost.
2. **Close the open ADRs.** 0008 and 0012 are sycophancy designs that the concept
   pipeline replaces rather than fixes. Mark them **Superseded by ADR 0014**. 0010
   (steering) gets moved to **Accepted**, because the new design depends on it and it
   has been used in 3 reported runs already. That is honest and it closes them.
3. **The sandbagging result is not being reported as it stands.** The labels are partly
   determined by which side the answer key sits on (cave rate 52% when the correct
   answer is (A), 28% when it is (B), z = 3.85). Say this before someone else finds it.
   Pull it from the deck and the talk outline. Then say that this confound is the reason
   for item 1, which makes the disclosure the start of the next experiment instead of
   just a retraction.
4. **Compute.** No new ask. Qwen2.5-7B on PAS2324 is what I am already running.

Item 4 got smaller, which is the strongest argument for item 1.

---

## Preserving the old work

A git tag does not preserve untracked files, and `scheming.py` is currently untracked.
Scheming is deferred, not dropped, so this still has to happen first. Explicit file
list, never `git add -A` (the `.gitignore` predates the `/docs/` rule).

```bash
git add src/dprobe/data/scheming.py src/dprobe/data/evalaware.py \
        src/dprobe/data/__init__.py tests/test_scheming_rollout.py \
        scripts/diagnose_surface_baselines.py scripts/letter_balance.py \
        scripts/scheming_one_question.py scripts/scheming_rollout_demo.py \
        CLAUDE.md
git commit -m "wip: scheming + evalaware modules, surface-baseline diagnostics"

git add docs/*.md .claude/docs/writing/ .claude/docs/adr/
git commit -m "docs: 07-16 sandbagging summary, 07-26 confound audit, critique, scope, ADR 0014"

git tag -a v0-sycophancy-sandbagging -m "5 weeks of sycophancy + sandbagging: 5 designs, 4 leaks, DiD steering. Labels contaminated by answer-key side; superseded by concept-derived directions (ADR 0014)."
git push origin sandbagging-did --tags

git switch -c concept-vectors
```

Push the tag. `/fs/scratch` on OSC is purgeable and already ate one probe.

Also: `rm .git/_writetest` (a leftover from a permissions check, harmless).

---

## The rule

Write the falsification criterion before the run, not the interpretation after it. Every
run gets a `prereg` field before it starts: what confirms, what kills, what I do in each
case. `write_meta` refuses to run without it.

And a second rule, added 2026-07-27: **no scope change without a line in the amendment
log and a stated counter-argument.** If I cannot write the counter-argument, I do not
understand the change well enough to make it.
