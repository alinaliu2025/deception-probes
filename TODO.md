# TODO

Scope: `docs/SCOPE.md`. Design: `docs/adr-0014-concept-derived-sandbagging-direction-DRAFT.md`.
Why the scope changed on 2026-07-27: the amendment log at the top of `SCOPE.md`.

**One thing:** does a direction built from a description of sandbagging, with no
behavioural label in its construction, steer the behaviour and detect it?

---

## → NEXT ACTION

**Commit, tag, branch.** Scheming is deferred, not dropped, and `scheming.py` is still
untracked. A tag does not preserve untracked files. Exact commands at the bottom of
`SCOPE.md`. Do this before touching anything else.

**Plumbing check: DONE 2026-07-27**, run
`2026-07-27T17-20-24Z_concept_concept_5b08b70_Alinas-MacBook-Pro-3`
(`--no-judge --skip-eval`, 0.5B, 120 records). Findings:

1. ✅ `extract_response_mean` offset is correct. `verify_response_span` shows pooling
   starting at `'The primary reason why'` / `'Mercury.'` / `'A catalyst increases the'`,
   straight after `<|im_start|>assistant\n`
2. ⏸ Judge survival rate: not measured, no backend yet
3. ⚠ **The positive pole does not sandbag at 0.5B.** "Mercury." is the correct answer to
   the question it was asked under a trait-eliciting system prompt. The model is not
   following the instruction at all. Expected at this size, and it is precisely the gap
   the judge filter exists to close. Re-check this at 7B: if the positive pole still
   refuses to sandbag there, the fixture systems are too weak and that is a week-1
   blocker, not a week-2 detail
4. ⚠ Ran on **CPU, not MPS**. `get_device()` checks `torch.backends.mps.is_available()`
   first and it returned false in the py3.9 venv. Worth 10 minutes before the next local
   run, though it does not affect OSC

**Bug found and fixed by this run:** `layer_sweep_concept` scored separation as the raw
gap between pole means, which is `||mu_pos - mu_neg||` and therefore grows with the
layer's residual norm. It picked layer 24 of 24. Now Cohen's d, which is scale-free. Two
regression tests in `tests/test_concept.py`.

---

## ⚠ Meeting: Thursday 2026-07-30

**The plumbing check above is the ONLY hard requirement before it.** Do not build the
judge backend properly this week. Instrument-verified plus a written design is a clean
ask; half-built shows nothing.

Agenda (full version in `SCOPE.md`):

- [ ] 1. Scope: concept-derived directions on sandbagging, 6 weeks, per ADR 0014. Lead
         with the fact that this replaced a scope written on Monday, and why. If the
         counter-proposal is scheming, take it, the design is written
- [ ] 2. Close ADRs 0008 and 0012 as **Superseded by ADR 0014**. Move 0010 to
         **Accepted**, it has been used in 3 reported runs
- [ ] 3. **The uncomfortable one.** The 07-16 sandbagging result isn't being reported as
         it stands. One table: cave rate 52% when the correct answer is (A) vs 28% when
         it's (B), z = 3.85, so part of the "held" class is failed compliance landing on
         the answer key by accident. Say it before someone else finds it. Then connect
         it to item 1. Prep this Wednesday, it's the item most likely to get dropped
- [ ] 4. Compute: no new ask. That is the strongest argument for item 1

The scope argument does **not** depend on any result. It stands on
`docs/critique-2026-07-26.md` and the z = 3.85 table.

---

## This week

- [ ] Commit untracked files, tag `v0-sycophancy-sandbagging`, push, branch to
      `concept-vectors`. **Do this first.**
- [ ] `rm .git/_writetest` (leftover from a permissions check)
- [ ] Move `docs/adr-0014-...-DRAFT.md` into `.claude/docs/adr/0014-...md` and drop the
      DRAFT suffix once the meeting signs off. Same for `adr-0013`, which has been
      sitting in `docs/` since it was written
- [ ] Plumbing check ← **next action, above**

## Week 1 build

- [ ] Expand `src/dprobe/data/fixtures/concept_sandbagging.json`: questions 12 → ~40,
      regenerate the 5+5 systems from `description` with an external LLM, review by
      hand, commit the result. Vary the **question domain**, not the phrasing. Poles
      must stay equal in count
- [ ] Wire a judge backend into `scripts/concept_vector.py` (currently `stub_judge`).
      LiteLLM proxy is the obvious one
- [ ] Hand-score 50 responses against the judge, record the agreement rate in
      `meta.json`. Under 0.85, rewrite the rubric before going further. This is the one
      live risk named in `SCOPE.md`
- [ ] Extraction run on 7B. Save `probe.npz`, scp it back from OSC
- [ ] `scripts/steer.py --control random --forced-choice` on the concept probe. Both
      passes. **Read the pre-registered decision table in ADR 0014 first. Do not
      deliberate after the fact, the numbers pick the outcome**
- [ ] Detection AUROC on the held-out DiD split, then the (A)/(B) split. That split is
      the headline

## Controls, before any number gets reported

- [ ] Opposing pole: steer with `−v`
- [ ] Prompt-style vector: `--no-judge --read prompt-final`. If it matches the real one,
      the pipeline bought nothing
- [ ] KL-matched steering control (infra list below)
- [ ] Permutation on the detection eval

## Infrastructure, one afternoon, retires 5 critique items permanently

- [ ] Three-way split in `evaluate.py: layer_sweep` (~15 lines)
- [ ] Bootstrap CIs on AUROC in `evaluate.py` (~20 lines)
- [ ] Persist activations to the run dir in `runlog.py` (~40 lines)
- [ ] `prereg` field required by `write_meta` (~5 lines)
- [ ] KL-matched steering control in `steer.py` (~30 lines)

## Done

- [x] Harsh review of the project (`docs/critique-2026-07-26.md`)
- [x] Scope contract (`docs/SCOPE.md`), amended 2026-07-27
- [x] Protocol walkthrough on one scenario (scheming, now deferred)
- [x] Fix 4 bugs in `scheming.py`
- [x] ADR 0014 draft + `src/dprobe/concept.py` and `scripts/concept_vector.py` sketches

---

## Not doing until 2026-09-06

In-context scheming and the Qwen3-8B download. Omission. Eval-awareness. The full 3x3
transfer matrix. The model ladder. Any new deception type built from behavioural labels.

New ideas go in the parking lot at the bottom of `SCOPE.md`, with a date. Not into the
branch. And per the second rule in `SCOPE.md`, no scope change without an amendment-log
line and a written counter-argument.
