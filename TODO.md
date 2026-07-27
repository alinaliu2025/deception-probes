# TODO

Scope: `docs/SCOPE.md`. Protocol: `docs/protocol-walkthrough-one-scenario.md`.
Answers to open questions: `docs/pilot-questions-answered.md`.

**One thing:** does a reasoning model represent in-context scheming, and can a probe
detect it in the cases where the CoT does not admit to it?

---

## → NEXT ACTION

**Start the Qwen3-8B download now**, on an OSC **login** node (Ascend, per
`scripts/osc/*.sbatch`). ~16GB, compute nodes have no internet, and it's the long
pole. It can run while you do anything else.

```bash
export HF_HOME=/fs/scratch/PAS2324/hf
source activate /fs/ess/PAS2324/dprobe-env-alina
hf download Qwen/Qwen3-8B
```

Then: **confirm thinking mode emits `<think>` blocks AND hidden states.**
30-minute interactive job, code in `pilot-questions-answered.md` Q2 step 3.
Write down three things:

1. Does a `<think>` block appear at all
2. `len(hidden_states)` — every layer number in ADRs 0001-0012 is meaningless
   on this model, it's deeper than Qwen2.5-7B
3. Peak GPU memory on a 1024-token generate — this sets `num_return_sequences`
   for the real pilot, and OOM is the likeliest thing to eat a day later

---

## ⚠ Meeting: Thursday 2026-07-30

**The model check above is the ONLY hard requirement before it.** Do not start the
pilot harness this week. Half-built shows nothing; instrument-verified plus a
written design is a clean ask.

Agenda (full version in `SCOPE.md`):

- [ ] 1. Scope: scheming only, 6 weeks, dropping sycophancy / sandbagging / omission
- [ ] 2. Close ADRs 0008, 0010, 0012 as **Superseded by scope change**
- [ ] 3. **The uncomfortable one.** The 07-16 sandbagging result isn't being
         reported. One table: cave rate 52% when the correct answer is (A) vs 28%
         when it's (B), z = 3.85, so part of the "held" class is failed compliance
         landing on the answer key by accident. Say it before someone else finds
         it. Prep this Wednesday, it's the item most likely to get dropped.
- [ ] 4. Compute ask: Qwen3-8B on PAS2324

The scope argument does **not** depend on pilot results. It stands on
`docs/critique-2026-07-26.md`.

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
