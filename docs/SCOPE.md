# SCOPE — in-context scheming, 2026-07-26 to 2026-09-06

This is the contract. It is the only document that says what I am working on. If
something is not in here, I am not doing it, and the way to change that is to edit
this file deliberately, not to drift.

Branch: `scheming`. Model: Qwen3-8B (thinking mode). Duration: 6 weeks.

---

## The one thing

**Does a reasoning model represent in-context scheming, and can a probe detect it in
the cases where the model's chain of thought does not admit to it?**

That second clause is the whole contribution. Apollo's own framing is that CoT
monitoring is a *lower bound* on scheming, because a model can scheme without
verbalising it. If that's true, a probe is the only instrument that can measure the
gap. So the headline figure is not "probe AUROC on scheming." It's **probe AUROC
split by whether the model said anything about it in its reasoning.**

If the probe holds up on the silent cases, that is a real result and it is not one
that CoT monitoring can produce.

---

## Why this design cannot repeat the last 4 weeks

Stated explicitly, because I have built 5 datasets that leaked and I need to know in
advance why this one is different.

| Past failure | Does it apply here? |
|---|---|
| Prompt-token leakage (`completion`, `framing`) | **No.** The label comes from what the model does, never from a property of the prompt. |
| Letter shortcut (`did`, answer-token arm) | **Structurally impossible.** The label is a multi-step covert action, not a token at a read position. There is no letter to match. |
| Answer-side bias, (A) vs (B) | **Structurally impossible.** No 2-option forced choice. |
| Content confounding (`behavioral`) | **Still live. This is the risk.** If scenario X always schemes and scenario Y never does, topic predicts the label again. Handled by the design the pilot selects (see below), plus the bag-of-words control before I trust any number. |
| Rollout starvation (`rollout`) | **Possibly solved.** Rollouts starved on sycophancy because the model was 99.8% committed per sample, so almost no question produced both classes. Scheming rates in the 5-40% band are exactly the regime where within-scenario rollouts work. The pilot measures this directly rather than assuming it. |

One risk remains live and I know which one it is. That's the difference from last time.

---

## Week 1: the pilot that can kill this

**Question:** does an 8B reasoning model scheme at all, and if so, is the variance
within scenarios or between them?

**Run:** the 6 reconstructed agentic scenarios in `scheming.py`, N = 20 samples each,
temperature 0.7, thinking enabled. For every sample record the full CoT, the final
action, whether the covert lever was taken, and whether the CoT mentions oversight,
monitoring, or being watched.

**Three numbers out:**

1. Covert-action rate, overall and per scenario.
2. Where the variance lives: within scenario (same scenario gives both outcomes across
   samples) or between scenarios (each one is roughly always or roughly never).
3. CoT verbalisation rate: how often the reasoning says the quiet part.

**Pre-registered decisions. I write these down now so the result forces a choice
instead of generating a sixth design.**

- **Covert rate < 5% overall** → an 8B does not scheme on these scenarios. Try 14B
  once. If still under 5%, this direction needs frontier-scale models I do not have,
  and I pivot. **I find this out in week 1, not week 6.**
- **5-40%, variance within scenarios** → rollout design. Label per sample, balance
  cave/hold within each scenario, content cancels in the class means.
- **5-40%, variance between scenarios only** → rollouts will starve. Use DiD with an
  oversight-present vs oversight-absent contrast, subtract within scenario.
- **Over 60%** → the scenarios are too leading and this is effectively instructed
  behaviour, which is the sandbagging mistake again. Tighten them and re-pilot once.

**Verbalisation is a second, independent gate.** If the CoT admits to scheming in over
90% of covert cases, there is no silent set and the headline question has no data.
That doesn't kill the project, but it changes the claim, and I need to notice it in
week 1 rather than discover it in the writeup.

---

## Weeks 2-6

2-3. Build the design the pilot selected. Run the controls **before** trusting any
number: permutation, bag-of-words on the full transcript, neutral-read. A design that
fails these gets fixed or dropped, not reported.

4. Probe: layer sweep with the corrected three-way split, bootstrap CIs on every cell.

5. The headline split: probe AUROC on the cases where the CoT verbalised, versus the
cases where it didn't.

6. Steering with KL-matched controls. Write up.

---

## Infrastructure: about 110 lines, once, and 5 of the critique's problems go away

This is the part that felt overwhelming and mostly isn't. These are one-time fixes to
shared code, not per-experiment work. Do them in week 1 alongside the pilot.

| Fix | Where | Lines | Retires |
|---|---|---|---|
| Three-way split (train / layer-select / test) | `evaluate.py: layer_sweep` | ~15 | Selecting the best layer on the test set and reporting that max |
| Bootstrap CIs on AUROC | `evaluate.py` | ~20 | Every bare point estimate in the project |
| Persist activations to the run dir | `runlog.py` | ~40 | Re-running GPU jobs to answer questions that are linear algebra on data I already computed |
| `prereg` field, required by `write_meta` | `runlog.py` | ~5 | Runs with no stopping rule |
| KL-matched steering control | `steer.py` | ~30 | Norm-matched controls that inflate the reported effect size |

Do not also fix `transfer_matrix` refitting on train+test. It only matters for the
cross-type study, which is out of scope.

---

## Explicitly out of scope until 2026-09-06

Not cancelled. Deferred, with a date. If I want to move something up here, I edit this
file and note why.

- Sycophancy, all 5 designs
- Sandbagging, both arms
- Omission
- Eval-awareness (`evalaware.py`) — and note the detection result is already published
  (Nguyen et al. 2507.01786), so this was never as novel as it felt
- The 3x3 cross-type transfer matrix
- The model ladder (0.5B → 14B)
- Any new deception type

---

## Parking lot

New ideas go here with a date. They do not go into the branch. Reviewed on 2026-09-06.

- (2026-07-26) Concept-derived directions à la Anthropic's emotion vectors: build the
  direction from a concept description rather than a behavioural outcome, so it never
  touches the label. Structurally immune to the circularity problem.
- (2026-07-26) Opposing-pole control instead of only a random direction.
- (2026-07-26) Dose-response check: vary cue strength in steps, confirm the probe score
  moves monotonically.
- (2026-07-26) Audit Apollo's public RoleplayDeception set with the leak protocol. The
  honest and deceptive completions differ in topic, not just honesty, so it probably
  fails bag-of-words. Worth a paper section if the main line leaves room.
- (2026-07-26) Nonlinear probe on saved activations, to check what linearity costs.

---

## The one meeting

I skip approval and just try things, which is why 3 ADRs have been stuck at Proposed
for weeks while results from them got reported anyway. The fix is not asking 12 times.
It's asking once, about scope. One agenda, one meeting, this week:

1. **Scope.** Scheming only, 6 weeks, dropping sycophancy / sandbagging / omission.
   Get a yes or a counter-proposal.
2. **Close the open ADRs.** 0008, 0010, and 0012 are all sycophancy and sycophancy is
   now out of scope. Mark them **Superseded by scope change**, not Accepted and not
   Rejected. That's honest and it closes them.
3. **The sandbagging result is not being reported.** The labels are partly determined
   by which side the answer key sits on (cave rate 52% when the correct answer is (A),
   28% when it's (B), z = 3.85). Say this before someone else finds it. Pull it from
   the deck and the talk outline.
4. **Ask for the compute.** Qwen3-8B thinking mode on OSC PAS2324.

---

## Preserving the old work

A git tag does not preserve untracked files, and `scheming.py` is currently untracked
and is the foundation of this branch. So commit first. Explicit file list, never
`git add -A` (the `.gitignore` predates the `/docs/` rule).

```bash
git add src/dprobe/data/scheming.py src/dprobe/data/evalaware.py \
        src/dprobe/data/__init__.py tests/test_scheming_rollout.py \
        scripts/diagnose_surface_baselines.py scripts/letter_balance.py \
        scripts/scheming_one_question.py scripts/scheming_rollout_demo.py \
        CLAUDE.md
git commit -m "wip: scheming + evalaware modules, surface-baseline diagnostics"

git add docs/*.md .claude/docs/writing/
git commit -m "docs: 07-16 sandbagging summary, 07-26 confound audit, critique, scope"

git tag -a v0-sycophancy-sandbagging -m "4 weeks of sycophancy + sandbagging: 5 designs, 4 leaks, DiD steering. Superseded by scope change to scheming."
git push origin sandbagging-did --tags

git switch -c scheming
```

Push the tag. `/fs/scratch` on OSC is purgeable and already ate one probe.

Also: `rm .git/_writetest` (a leftover from a permissions check, harmless).

---

## The rule

Write the falsification criterion before the run, not the interpretation after it.
Every run gets a `prereg` field before it starts: what confirms, what kills, what I do
in each case. `write_meta` refuses to run without it.
