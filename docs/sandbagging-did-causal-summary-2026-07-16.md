# Session summary: sandbagging DiD goes causal, 2026-07-15 → 07-16

Everything built, run, and learned this working session. The headline: porting the
difference-of-differences (DiD, ADR 0012) design to sandbagging, then switching the
probe estimator from `lr` to `mms`, produced the **first causally-sufficient
sandbagging direction** in the project — and a DiD cross-type transfer matrix that
reproduces the asymmetric syco/sand transfer on the trustworthy probes.

Companion docs: `session-summary-2026-07-12-16.md` (the rollout-arm campaign this
continues), `sandbagging-pressure-arms-summary-2026-07-14.md` (pressure arms),
`.claude/docs/adr/0012-sycophancy-difference-of-differences.md` (the design ported).
Figures: `results/figures/fig1_steering_lr_vs_mms.png`,
`fig2_crosstype_transfer.png`, `fig3_detection_layer_sweep.png` (regenerable from
git-tracked metas).

Branch: `sandbagging-did`. Model: Qwen2.5-7B-Instruct unless noted. All OSC runs
are the user's; nothing was executed on OSC from this session directly.

---

## 1. What was done, in order

1. Reviewed the docs, laid out the sandbagging-DiD ladder (fixture → 29 → 400 →
   3000), following Jack's rungs.
2. Mac fixture smoke (0.5B, incentive arm) → single-class collapse (0 caved / 20
   held). Exposed an unhandled empty-filter crash; fixed it (clean abort).
3. Built `scripts/osc/train_did.sbatch` (7B default, **refuses to run without
   `--max-examples`** so a bare `--source factual` can't load the 7k pool).
4. Rung 1 (29q, incentive, 7B): 0 caves, clean abort. Pipeline verified end to end
   (gate 29/29, 0 unparsed, 7B not silently defaulted to 0.5B).
5. Fixed the port wart: `sycophancy_base_rate` → `cave_base_rate` in both
   `did_filter` and `behavioral_filter` (the key is shared with sandbagging).
6. Rung 2 (400q, both arms). Instructed trained; incentive caved 1/388 → single-
   class crash → extended the guard to a floor of 4 examples/class.
7. Steered the **lr** instructed probe → **null** (detection ≠ causation, again).
8. Diagnosed the null vs Jack: estimator (`lr` vs `mms`) and layer (25 vs 19).
9. Ported **type-aware steering** to this branch (was only on `sandbagging-rollout`).
10. Refit the instructed arm as **`mms`** → steered → **causal** (0 → 0.72).
11. Built **`scripts/crosstype_did.py`** (DiD transfer in arrow-space), ran the 2×2.
12. Recovered the sycophancy DiD probe from Jack's purged scratch; made 3 figures.

## 2. Runs and results

| run dir | what | key numbers |
|---|---|---|
| `...T19-52-40Z_..._e029ac9_Mac` | fixture smoke, 0.5B, incentive | single-class (0/20) → crash, then clean-abort fix |
| `...T20-06-18Z_..._396810f_p0310` | rung 1, 29q incentive | 0 caved / 29 held → clean abort (`no_examples`) |
| `...T21-00-40Z_..._be3f1a2_p0310` | rung 2, 400q **instructed lr** | 160/228, balanced 320, **pf 0.931 @L25**, at 0.996 @L21, gap +0.065 |
| `...T21-25-04Z_..._e3cf235_p0310` | rung 2, 400q incentive | **1 caved / 387 held (0.26%)** → clean abort (`too_few`) |
| `...T23-46-27Z_steer_lr_54a37de_p0306` | steer lr, L25, instructed | **NULL**: add 0.0 at every usable α; random ≈ 0; ablate null |
| `...T03-49-37Z_..._b271758_p0306` | 400q **instructed mms** | 160/228, balanced 320, **pf 0.845 @L20**, at 0.888 @L20, gap +0.043 |
| `...T04-03-38Z_steer_mms_5e17417_p0306` | steer mms, L20, instructed | **CAUSAL**: add 0 → 0.39 → 0.46 → **0.72** @α1.0 (parse intact); random flat 0.013; **ablate 1.0 → 0.835** vs random 0.985 |
| `...T01-51-38Z_sycophancy_mms_cb29054_p0303` | Jack's syco DiD probe (mms L19) | recovered from `/fs/scratch`; probe.npz copied to project storage |
| `...T05-11-24Z_crosstype_did_mixed_9141814_p0330` | DiD 2×2 transfer | matrix below |

**Steering (add pass, α = 0, .25, .5, 1.0; parse-rate intact through 1.0):**

- lr @L25: probe `[0, 0, 0, 0]`, random `[0, 0, .008, .013]` → inert.
- mms @L20: probe `[0, .394, .46, .72]`, random `[0, 0, .003, .013]` → causal, ~55×
  the random control at α=1.0. Ablate: probe drops caving 1.0→0.835, random 0.985 —
  the **first necessity nudge** in the project (nobody's probe, Jack's included, had
  moved the ablate needle).

**DiD cross-type transfer AUROC** (row = probe, col = dataset's prompt-final arrows):

|                | syco data (n=44) | sand data (n=254) |
|----------------|:---:|:---:|
| **syco probe** (mms L19) | 0.876 | 0.559 |
| **sand-inst probe** (mms L20) | 0.839 | 0.709 |

direction cosine (syco vs sand, cross-layer, indicative) = **−0.13**.

## 3. The interpretation

- **`lr` detects, `mms` drives.** Same 400q data, same ~L20 depth, same harness —
  the only change from the null run to the causal run was the estimator. `lr` learns
  a discriminative hyperplane normal (great AUROC 0.931, but free to lean on
  nuisance dimensions); `mms` is the literal difference-of-means axis the behaviour
  travels on, which is what steers. Detection AUROC and causal effect are orthogonal
  axes — the campaign's standing lesson, now with a clean positive to pair against
  the earlier nulls.
- **Asymmetric transfer, reproduced on trustworthy probes.** The sandbagging
  direction detects sycophancy caving (0.839) but sycophancy does **not** detect
  sandbagging (0.559 ≈ chance); directions near-orthogonal. Same pattern as the
  rollout matrix (sand→syco 0.95 / syco→sand 0.52), now on DiD probes. Reading: the
  instructed-sandbagging direction is a **generic "committing to a known-wrong
  answer / capitulation" axis** present in sycophancy caving too; sycophancy's
  direction carries something type-specific (and, per Jack, causal) that sandbagging
  data lacks.
- **The syco→syco 0.876 diagonal reproduces Jack's 0.878** — a pipeline sanity check
  that validates the crosstype plumbing and probe loading.
- **The counterintuitive cell** (sand probe scores higher on syco data 0.839 than
  its own 0.709) is partly the n=44 syco cell (wide error) and partly the generic-
  capitulation reading; not a labelling bug (verified against the raw meta).

## 4. Code changes (branch `sandbagging-did`)

- `scripts/train_one.py` (did path): clean abort with meta + trace when the filter
  leaves **< 4 examples/class** (was an unhandled `extract_arrows`/sklearn crash;
  records `aborted: too_few_examples_after_filter`).
- `scripts/osc/train_did.sbatch` (new): 7B default, `--max-examples` guard.
- `src/dprobe/data/sycophancy.py`: `sycophancy_base_rate` → `cave_base_rate` (both
  filters; legacy metas keep the old key, noted in-comment).
- `src/dprobe/steer.py`: `build_items` is now **type-aware** — sandbagging swaps the
  SYSTEM prompt (`calm_system`) for the calm side; sycophancy still swaps the user
  turn (`neutral_user`). Steering the wrong turn was a silent category error.
- `scripts/steer.py`: reads `probe.deception_type` (was hardcoded `"sycophancy"`);
  added `--pressure {instructed,incentive}`; records the arm in meta.
- `scripts/osc/steer.sbatch` (new): `--forced-choice --control random`, `--probe` guard.
- `scripts/crosstype_did.py` + `scripts/osc/crosstype_did.sbatch` (new): DiD transfer
  in **arrow-space** (a did probe can't be scored on raw pressured snapshots — ADR
  0012); `--max-examples` caps **before** the belief gate.
- Tests: `tests/test_sandbagging_did.py` +3 (build_items routing);
  `tests/test_crosstype_did.py` (new, per-layer readout + parse_spec).
- `results/figures/fig1..3` (new).

**NOTE — not yet run/committed cleanly:** `pytest` was NOT run this session (the
tooling env lacked torch); **run `pytest -q` locally before trusting the ports.**
Several edits (steer port, crosstype, figures, the `cave_base_rate` rename) may
still be uncommitted — commit with an explicit file list (the branch `.gitignore`
predates the `/docs/` rule; do **not** `git add -A`), and flag the two-line edits to
Jack's files on push.

## 5. Concepts clarified

- **mms vs lr for steering**: difference-of-means is the representational axis;
  classifier normal is discriminative. Equal detection, very different causality.
- **train-best-layer vs fresh-test AUROC**: 0.845 (best-of-29 layers, held-out
  *within* the 400q pool) vs 0.709 (same fixed L20 on the fresh ARC test split, no
  layer cherry-pick). The drop is layer-selection optimism + true out-of-sample, not
  a regression bug; report the fresh number.
- **OSC storage**: `/fs/scratch` is purgeable (the syco probe vanished from it and
  was recovered from Jack's copy via an in-cluster `cp` to `/fs/ess` project
  storage). Keep probe.npz off scratch; only `meta.json` travels via git.
- **`--max-examples` placement**: capping before the filter actually bounds the
  expensive gate; capping after is cosmetic (fixed in crosstype).

## 6. Open items / next steps

1. **Narrow fix for the transfer cells**: bootstrap CIs (the 0.839 vs 0.709
   ordering is likely within noise at n=44); report an **lr** detection number on the
   fresh test split alongside the **mms** steering number (one probe is currently
   conflating detection / generalization / causation).
2. **The real interpretive fix — a non-instructed arm.** The causal, transferring
   direction is from the *instructed* arm, so it can't yet separate sandbagging from
   instruction-following. Build the middle-ground / incentive prompt, pilot at 400q
   to check the cave rate clears the floor, then steer + transfer it. If it also
   steers and transfers to sycophancy, the generic-capitulation reading is confirmed.
3. **Harden the causal claim**: permute-trained direction as a second steering
   control; seed repeat (is 0.72 stable?); a second mid-layer steer. Persisting the
   arrows would make these (and layer sweeps) free — proposed, **not yet built**.
4. Add the incentive/middle-ground row to the transfer matrix; later the
   omission/concealment third type for the full 3×3.
5. Model ladder (0.5B → 1.5B → 14B) once the 7B story is airtight.
6. **ADR 0012 is still Proposed** — bring the causal result to the sign-off
   conversation (Dr. Zhu / Jack / Xudong) before reporting.

## 7. Meta-lessons

- The controls did their job again: the lr null, the random steering baseline, and
  the syco→syco sanity diagonal each either protected or validated a claim. The
  causal result only means something *because* the lr version was null in the same
  harness.
- Detection-optimal and causation-optimal designs differ — and so do detection-
  optimal and causation-optimal **estimators**. Know which question a probe is built
  for before reading its number.
- A surprising cell (sand→syco > sand→sand) is a prompt to check n and add error
  bars, not to assume a bug — the raw meta is the arbiter.
