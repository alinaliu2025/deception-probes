# dprobe paper todo — v2 (2026-07-08)

**Goal:** a publishable paper whose central result is the 3×3 cross-type transfer
matrix (sycophancy, sandbagging, omission), plus how transfer changes with model
scale.

This replaces the v1 todo, which was written before Jack's June–July work and no
longer matched the repo. Sources: run catalog (`.claude/docs/run-catalog.md`),
ADRs 0001–0010, `docs/HPC.md`, group notes doc.

---

## Where things actually stand

- Pipeline runs end to end, locally and on OSC (project PAS2324), up to Qwen 7B.
  25+ logged runs with per-run `meta.json`.
- **Sycophancy:** three dataset designs were built and rejected — `completion` and
  `framing` leak (AUROC ≈ 1.0 at layer 1; permutation control fails), `behavioral`
  is content-confounded (neutral-read control hits AUROC 1.0). The `rollout` +
  `--source factual` design is the current candidate, **pending team sign-off**
  (ADRs 0008–0009). Steering validation has started (ADR 0010).
- **Sandbagging and omission:** still the 12-item seed sets. No real data.
- **No trustworthy transfer matrix exists yet.** The paper's central artifact is
  unbuilt because two of three datasets don't exist and the third isn't signed off.

The lesson from June: dataset construction is where this project fails or
succeeds. Every design so far that looked good leaked. Assume the next one leaks
too until the controls say otherwise.

---

## Standing rules (apply to every dataset, all three types)

A dataset is usable only if it passes all of these:

1. **Permutation control:** shuffle labels, refit → AUROC ≈ 0.5. Higher means leakage.
2. **Neutral-read control:** extract activations under a neutral prompt → probe
   should *not* separate the classes from content alone.
3. **Capability filter** (sandbagging only): keep only items the model answers
   correctly under the neutral prompt.
4. **No scenario overlap** across types or across train/test splits (split by
   scenario family, not row).
5. **Frozen and versioned:** hash each split; record counts in a manifest. Never
   regenerate silently.

---

## Step 1 — first honest transfer matrix on small models (next ~2 weeks)

This is the highest-leverage step: it tests whether the transfer story exists at
all before more OSC spend.

- [ ] **Sign off the rollout+factual sycophancy design** (team meeting with Jack
      and Dr. Zhu). If rejected, decide the replacement in the same meeting.
- [ ] **Build a minimal sandbagging set** (~200 usable pairs). Cheapest route per
      group notes: prompt-based underperformance on MCQs with the capability
      filter. Verify the sandbagging behavior actually appears before extracting
      activations. (Password-locked models — FelixHofstaetter/mistral-7b,
      redwoodresearch/math_pwd_lock — are the stronger alternative but tie us to
      Mistral/DeepSeek; decide in the same meeting.)
- [ ] **Build a minimal omission set** (~200 pairs). Start from the Apollo
      deception-detection repo's insider-trading concealment scenario (clean
      ground-truth labels) rather than writing scenarios from scratch.
- [ ] Run both new sets through the standing rules above.
- [ ] **Run `scripts.compare` on Qwen 0.5B and 7B** → first 3×3 matrix + direction
      cosines that we can believe.

**Done when:** a matrix exists where every input dataset passed its controls.
Rough is fine. Show it at group meeting before anything below.

---

## Step 2 — make the numbers paper-grade

- [ ] Multiple seeds (5): re-split, refit, report mean ± 95% CI on every number.
- [ ] Layer selection rule, fixed once: best layer on the *train type's val set*,
      then frozen. Never picked on test.
- [ ] Extend `runlog.py` so every run records: config, git SHA, dataset hash, seed.
- [ ] Leave-one-type-out probe (train on two types, test on the third) and
      train-on-all probe. These test the "shared deception direction" claim.
- [ ] Asymmetry test: is cell (i,j) vs (j,i) significant, paired over seeds?

**Done when:** one command reproduces the matrix, cosine heatmap, and
leave-one-out table with CIs.

---

## Step 3 — scale on OSC

Model ladder per ADR 0002: 0.5B → 1.5/3B → 7B (formal) → 14B (stretch), single
GPU, switched via `DPROBE_MODEL`, no code edits.

- [ ] **Write `scripts/osc/compare.sbatch`.** ADR 0002 plans it and `docs/HPC.md`
      references it, but it is not in the repo yet. Until then runs are
      interactive-only, which wastes GPU allocation on idle time.
- [ ] Repeat the Step-2 outputs at each ladder size.
- [ ] Scaling plot: diagonal and off-diagonal AUROC (and direction cosines) vs
      model size, with CIs.

**Open decision (flag, don't assume):** Qwen-only ladder vs adding a second
family (group notes lean Gemma 2; the password-locked sandbagging models are
Mistral). One family is cleaner for scaling claims; a second family is an
external-validity check. Decide with Dr. Zhu when Step 1 results exist.

---

## OSC setup for Alina (account + PAS2324 access already exist)

Full reference: `docs/HPC.md` on Jack's branch. Condensed first-time path:

1. **Log in:** browser → `ondemand.osc.edu`, or terminal →
   `ssh <osc-username>@pitzer.osc.edu`. Duo two-factor required.
2. **Confirm project access:** `OSCfinger $USER` → should list PAS2324.
   If not, ask Dr. Zhu to add you at my.osc.edu → Project → Members.
3. **GitHub auth on OSC** (one time): `ssh-keygen -t ed25519`, then add
   `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH keys.
4. **Clone the repo** to persistent project storage, not `$HOME` (small quota):
   ```bash
   cd /fs/ess/PAS2324
   git clone git@github.com:alinaliu2025/deception-probes.git dprobe-alina
   cd dprobe-alina && git switch sycophancy_v2
   ```
5. **Python env** (one time). Make your own — the shared env's editable install
   points at Jack's clone:
   ```bash
   module load miniconda3
   conda config --remove channels defaults
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   conda create --prefix /fs/ess/PAS2324/dprobe-env-alina python=3.10
   export PYTHONNOUSERSITE=True
   source activate /fs/ess/PAS2324/dprobe-env-alina   # NOT `conda activate`
   pip install -e ".[dev]"
   ```
6. **Model cache off $HOME** (add to `~/.bashrc`):
   ```bash
   export HF_HOME=/fs/scratch/PAS2324/hf
   ```
7. **Smoke test on a GPU node:**
   ```bash
   sinteractive -A PAS2324 -g 1
   python -m pytest -q
   DPROBE_MODEL=Qwen/Qwen2.5-7B-Instruct python -m scripts.train_one --type sycophancy
   ```
8. **Formal runs:** batch via `sbatch scripts/osc/compare.sbatch` once that script
   exists (Step 3). Monitor with `squeue -u $USER`; cancel with `scancel <jobid>`.
9. **Storage rules:** `$HOME` = tiny quota; `/fs/scratch/PAS2324` = fast, purged
   ~90 days; `/fs/ess/PAS2324` = persistent. Copy results you care about to ess.

All compute charges go to PAS2324 (lab funds). Interactive Desktop sessions are
SLURM jobs too — pick a **GPU-enabled** node type or you silently get CPU.

---

## Step 4 — controls and ablations (after the scaled matrix exists)

Controls, all required:

- [ ] Random-direction floor (AUROC ≈ 0.5).
- [ ] Permutation control per cell (already standard — keep it).
- [ ] Topic control: a bag-of-words text classifier on the same data. If plain
      text transfers as well as activations, the interp claim collapses.
- [ ] Confound check: cosine of deception direction vs refusal, sentiment, and
      length directions.

Ablations, one axis each: probe family (mms vs lr — note mms was near-chance at
0.5B, run #1–4), token aggregation, train-set size curve.

Causal validation: continue the ADR 0010 steering line — add/subtract the
direction, measure the change in deceptive behavior. This is what lifts the
paper above "we trained a classifier."

---

## Step 5 — robustness

- [ ] Held-out scenario families within each type.
- [ ] Different chat template / system prompt; report the AUROC delta.
- [ ] Paraphrased eval prompts.

---

## Step 6 — write the paper

- [ ] Figures first: Fig 1 transfer matrix, Fig 2 direction cosines, Fig 3
      scaling curve. Captions must stand alone.
- [ ] Draft Method and Results first; Intro and Related Work last. Position
      against: Goldowsky-Dill/Apollo 2502.03407, Kumar 2605.27958 (predicts
      subspaces grow *disjoint* with scale — our H-scaling should cite it, not
      assume transfer improves), Vennemeyer 2509.21305 (separability *within*
      sycophancy), the taxonomy paper 2604.04788.
- [ ] Honest limitations section: judge-labeled data, English-only, ≤14B, three
      types out of a larger taxonomy.
- [ ] Use Liars' Bench (2511.16035) as held-out evaluation only, never training.
- [ ] Review: Jack → Dr. Zhu → arXiv → workshop (SoLaR or similar); main track
      only if the scaling result is strong.

---

## Open decisions (resolve at group meeting, in this order)

1. Rollout+factual sycophancy design: approve or replace. **Blocks everything.**
2. Sandbagging source: prompt-based on Qwen vs password-locked Mistral models.
3. Model family for scaling: Qwen ladder alone vs + Gemma 2.
4. Venue (sets how many of Step 4–5 are mandatory).
