# TODO

Scope: `docs/SCOPE.md`. Protocol: `docs/protocol-walkthrough-one-scenario.md`.
Answers to open questions: `docs/pilot-questions-answered.md`.

**One thing:** does a reasoning model represent in-context scheming, and can a probe
detect it in the cases where the CoT does not admit to it?

---

## → NEXT ACTION

**Confirm Qwen3-8B thinking mode emits `<think>` blocks AND hidden states, on OSC.**
30-minute interactive job. Code is in `pilot-questions-answered.md` Q2 step 3.
Write down `len(hidden_states)`, because every layer number in ADRs 0001-0012 is
meaningless on this model.

---

## This week

- [ ] Commit untracked files, tag `v0-sycophancy-sandbagging`, push, branch to
      `scheming`. Exact commands at the bottom of `SCOPE.md`. **Do this first, a tag
      does not preserve untracked files and `scheming.py` is untracked.**
- [ ] `rm .git/_writetest` (leftover from a permissions check)
- [ ] Download `Qwen/Qwen3-8B` on an OSC **login** node (compute nodes have no internet)
- [ ] Confirm thinking mode + hidden states ← **next action, above**
- [ ] Book the one meeting. Agenda is 4 items, in `SCOPE.md`. The uncomfortable one is
      item 3: tell them the sandbagging result isn't being reported and why.

## Week 1 build

- [ ] Write ~70 free-form scenarios using `FREEFORM_SCENARIOS[0]` as the template.
      Vary the **domain**, not just the names. Rule for writing them: the environment
      may contain an editable oversight config; nothing may name the covert action or
      imply permission. (`pilot-questions-answered.md` Q5.)
      Start with 10 to de-risk the harness, then scale.
- [ ] Write `scripts/pilot_scheming_rate.py`:
      1. Load Qwen3-8B, thinking on, confirm hidden states
      2. Calm gate, N=20 per scenario, drop scenarios that scheme when calm
      3. Pressured rollouts, N=20 per scenario, store full CoT + tool calls
      4. Grade `covert_action` from final file state, `verbalised` by keyword,
         `denied` from the follow-up turn
      5. Print: covert rate overall + per scenario, within-vs-between variance,
         verbalisation rate, required-scenario-count
      6. Write `prereg` into `meta.json` **before** the run
- [ ] Run the pilot. Read the decision rule in `SCOPE.md`. Do not deliberate, the
      numbers pick the design.

## Infrastructure, one afternoon, retires 5 critique items permanently

- [ ] Three-way split in `evaluate.py: layer_sweep` (~15 lines)
- [ ] Bootstrap CIs on AUROC in `evaluate.py` (~20 lines)
- [ ] Persist activations to the run dir in `runlog.py` (~40 lines)
- [ ] `prereg` field required by `write_meta` (~5 lines)
- [ ] KL-matched steering control in `steer.py` (~30 lines)

## Done

- [x] Harsh review of the project (`docs/critique-2026-07-26.md`)
- [x] Scope contract (`docs/SCOPE.md`)
- [x] Protocol walkthrough on one scenario
- [x] Fix 4 bugs in `scheming.py`: covert-always-(A), lever named in prompt,
      CoT-suppressing `DID_SUFFIX`, no free-form design

---

## Not doing until 2026-09-06

Sycophancy. Sandbagging. Omission. Eval-awareness. The 3x3 transfer matrix. The model
ladder. Any new deception type.

New ideas go in the parking lot at the bottom of `SCOPE.md`, with a date. Not into the
branch.
