# `dprobe` — Cross-Type Deception Transfer: Path to a Publishable Paper

**Main goal:** Find linear probes that detect different *types* of deception (sycophancy, sandbagging, omission/concealment) in LLMs, and characterize how well a probe trained on one type detects another — the **cross-type transfer matrix** as the central artifact.

**Target:** publishable research paper. **Timeline:** open-ended. **Current state:** package works on small models (Qwen 0.5B locally).

**How to read this:** phases are roughly sequential, but Phase 1 (data) and Phase 2 (methodology hardening) should run in parallel — data is the bottleneck, methodology hardening is cheap. Sections marked ⭐ are the experiment/analysis depth you asked for. Each task has a **Done when** so you know when to stop.

---

## Phase 0 — Lock the contribution (do this first, ~1 week)

The single most common reason interp papers don't land is a fuzzy claim. Nail this before running anything expensive.

- [ ] **Write a one-paragraph contribution statement.** The novel claim is *systematic* cross-type transfer + how it scales. Prior work (Goldowsky-Dill/Apollo, LASR, Cundy & Gleave, Parrack) mostly probes one deception setting; your artifact is the N×N transfer matrix across *distinct deception types* and its behavior across model scale. Write it in 4 sentences: problem, gap, what you do, what you find (leave the finding as a hypothesis for now).
- [ ] **Operationally define each deception type.** One precise definition + 2 canonical examples each. These definitions become your labeling spec.
  - *Sycophancy*: model changes/asserts a claim to match the user's stated belief or preference against its own "knowledge."
  - *Sandbagging*: model deliberately underperforms on a task it is capable of (capability-conditioned).
  - *Omission/concealment*: model withholds relevant true information it "knows," without asserting a falsehood.
- [ ] **State the core hypotheses explicitly** (pre-registration-style, so results aren't post-hoc):
  - H1: Within-type probes achieve high AUROC (diagonal ceiling).
  - H2: Off-diagonal transfer is above chance but below diagonal (partial shared signal).
  - H3: Transfer is **asymmetric** (e.g., a probe trained on the "clearest" type transfers better than the reverse).
  - H4: A shared "deception direction" exists — per-type probe directions have above-chance cosine similarity, and a leave-one-type-out probe generalizes.
  - H5 (scaling): transfer strength changes monotonically with model size.
- [ ] **Pick a target venue and read its bar.** Realistic ladder: arXiv preprint → interp/safety workshop (e.g., a NeurIPS/ICLR workshop, SoLaR) → potentially main-track. Note the page limit, whether code release is expected, and the reproducibility checklist. This determines how many ablations you actually need.
- [ ] **Define the minimum result that is publishable.** Write the sentence you'd need to be true. If it's "we present the first cross-type transfer matrix across ≥3 deception types on ≥3 model scales, with X," everything below is scoped to serve that sentence.

**Done when:** you have a half-page doc with the claim, definitions, hypotheses, venue, and MVP result. Send it to Dr. Zhu / Jack for a sanity check before building datasets.

---

## Phase 1 — Datasets (the real bottleneck) ⭐

You need, **for each of the 3 types**, matched honest-vs-deceptive contrast data. Quality and cleanliness here determine whether the transfer matrix means anything.

### 1a. Design the data schema (once)
- [ ] Fix a single record schema across all types: `{id, type, scenario, prompt, honest_completion, deceptive_completion, label, source, split, on_policy: bool, meta{}}`.
- [ ] Decide **on-policy vs off-policy** per type. On-policy (the model's own generations, then labeled) is stronger for a paper and avoids the "you just detect a stylistic artifact of templated text" critique. Aim for on-policy for the primary results; keep off-policy as a controlled comparison.
- [ ] Decide **positive/negative construction**: paired contrast (same scenario, honest vs deceptive) is best because it controls topic/content. This is what makes the probe learn *deception*, not *topic*.

### 1b. Build each type's dataset
For **sycophancy**:
- [ ] Source scenarios where a correct answer exists and a user asserts a wrong belief. (Feedback-sycophancy and answer-sycophancy templates; the Anthropic sycophancy eval sets and the Apollo/roleplay scenarios are reasonable seeds — reuse published sets where licensing allows, cite them.)
- [ ] Generate matched honest (holds correct answer) vs deceptive (caves to user) completions.

For **sandbagging** (you already have `phase0_sandbagging_transfer.py` with capability-filtered contrast pairs — reuse it):
- [ ] **Capability-filter first**: keep only items the model answers correctly under a neutral prompt (so a later wrong answer is *deliberate* underperformance, not incompetence). This is essential and easy to get wrong.
- [ ] Build the deceptive arm by prompting/eliciting underperformance on the capability-passing subset.

For **omission/concealment**:
- [ ] Build scenarios with a known relevant fact + an incentive to withhold. Honest = discloses; deceptive = withholds while not lying outright. Hardest to label — budget extra time.

### 1c. Label, validate, decontaminate
- [ ] **Validate labels** with an independent judge (LLM-judge with a rubric, plus a hand-audited sample of ≥100/type). Report inter-rater/judge agreement in the paper.
- [ ] **Decontaminate across types**: ensure scenarios/content don't overlap between types, or transfer will be inflated by shared topic rather than shared deception signal. Deduplicate near-duplicates (embedding similarity threshold).
- [ ] **Balance** class ratios and type sizes; record counts.
- [ ] **Split** train/val/test per type with **no scenario leakage** across splits (split by scenario family, not by row).
- [ ] **Freeze and version** datasets (hash each split; store under `data/` with a manifest). Never regenerate silently.

**Done when:** 3 versioned, decontaminated, judge-validated contrast datasets with clean splits and a manifest table (counts, agreement, on/off-policy) ready to drop into a paper appendix.

---

## Phase 2 — Harden the probe pipeline to paper grade ⭐

Your package runs; now make every run reproducible, seeded, and statistically honest. Do this on Qwen 0.5B before scaling — it's cheap to iterate.

### 2a. Activation extraction
- [ ] Confirm extraction site: residual stream (`hidden_states`) at **every layer**. Cache to disk keyed by `{model, dataset_hash, layer, token_agg}`.
- [ ] Implement/confirm **token aggregation** options and treat the choice as an ablation later: last token, mean over response tokens, last-of-response. Default to last response token but keep the switch.
- [ ] Use `bf16`, `torch.no_grad()`, deterministic seeds; verify extraction is idempotent (same input → same activations).

### 2b. Probe fitting
- [ ] Keep both probe families: **difference-of-means** (mass-mean direction) and **logistic regression** (with L2; sweep C on val). Optionally add LDA. Treat probe family as an ablation axis.
- [ ] Standardize features (fit scaler on train only).
- [ ] Save each probe as `{direction, bias, scaler, layer, meta}` so it's a portable object you can apply to *any* activation set (this is what makes the transfer matrix trivial to compute).

### 2c. Evaluation & statistics (the part reviewers check)
- [ ] Primary metric **AUROC**; secondary: accuracy at a val-chosen threshold, and a calibration check.
- [ ] **Multiple seeds** (≥5): reshuffle splits / re-fit; report mean ± 95% CI everywhere. A transfer matrix without CIs is not publishable.
- [ ] **Layer sweep** producing an AUROC-vs-layer curve per (train-type, eval-type). Pick the reporting layer by a fixed, pre-registered rule (e.g., best layer on the *train type's val set*, then frozen — never picked on test).
- [ ] Log every run (config, git SHA, dataset hash, seed) — Weights & Biases or a flat CSV + Hydra/JSON config. Reproducibility is a review criterion.

**Done when:** on Qwen 0.5B you can run one command that produces a within-type AUROC ± CI and a layer curve for all 3 types, fully seeded and logged.

---

## Phase 3 — Scale to larger models on OSC ⭐

You're on small models only; the scaling axis is a core part of the paper (H5).

- [ ] **Choose a clean scaling ladder** in one model family to avoid confounds: e.g., Qwen 0.5B → 1.5B → 7B → 14B (→ 72B if A100 memory allows). Same tokenizer/family = cleaner scaling story.
- [ ] **Activation storage plan.** Per-layer activations for large models × 3 datasets get big fast. Estimate `n_examples × n_layers × d_model × 2 bytes` per model; decide what to cache vs recompute; use scratch storage on OSC and clean up.
- [ ] **SLURM scripts** (you've done this): one job per (model, dataset) extraction; request A100(s), set `--mem`/`--time` from your estimate, `device_map="auto"` for the 14B+.
- [ ] Verify numerical parity: a probe re-fit from cached vs freshly extracted activations gives the same AUROC (guards against caching bugs).
- [ ] Extract activations for **all (model × type)** combos and store with manifest. This is the substrate for the whole results section.

**Done when:** you have cached, verified activations for every (model, type) on the ladder, with a storage manifest.

---

## Phase 4 — The transfer matrix (central artifact) ⭐⭐

This is the paper. Build it carefully and give it the most rigor.

### 4a. Core matrix
- [ ] For each model: fit a probe on **train type i** (best layer per 2c rule), evaluate its AUROC on **eval type j** for all i, j → an N×N matrix (N = 3).
  - **Diagonal** = within-type ceiling.
  - **Off-diagonal** = cross-type transfer.
- [ ] Compute each cell as **mean ± 95% CI over seeds.**
- [ ] Build the **heatmap figure** (annotated with values + CIs). This is likely your Figure 1.

### 4b. Generalization variants
- [ ] **Leave-one-type-out probe:** train on the other two types pooled, test on the held-out type. Tests whether a "general deception probe" transfers to an unseen type (H4).
- [ ] **Train-on-all probe:** upper bound for a unified detector.
- [ ] **Per-type-vs-general comparison:** does pooling help or hurt each type?

### 4c. Direction geometry (the mechanistic evidence for H4)
- [ ] Compute **cosine similarity between per-type probe directions** (at a matched layer) → a 3×3 similarity heatmap. High similarity = shared deception subspace; near-orthogonal = type-specific.
- [ ] Optionally: project all activations onto the leave-one-out "general direction" and check separation per type.

### 4d. Cross-model view (scaling, H5)
- [ ] Stack the transfer matrix across model sizes → show how diagonal and off-diagonal AUROC (and direction cosine similarity) trend with scale.
- [ ] Summarize with a single scaling plot: mean off-diagonal transfer vs model size, with CIs.

### 4e. Statistical treatment of the story
- [ ] Test whether **asymmetry** (cell ij vs ji) is significant (paired over seeds).
- [ ] Report a chance baseline (0.5 AUROC) and mark which cells clear it with CIs.

**Done when:** you have the annotated matrix heatmap(s), the direction-similarity heatmap, the leave-one-out results, and the scaling plot — each with CIs and a one-sentence takeaway.

---

## Phase 5 — Ablations, baselines & controls ⭐

Reviewers reject "you just detected a confound." Pre-empt every alternative explanation.

**Baselines / controls (do all of these):**
- [ ] **Random-direction baseline:** AUROC of a random unit direction (should be ~0.5). Sets the floor.
- [ ] **Shuffled-label control:** refit with permuted labels → should collapse to chance. Confirms no leakage.
- [ ] **Content/topic control:** train a probe to predict *topic* or *scenario family* from the same activations; show your deception probe isn't just reading topic. Alternatively, a bag-of-words/text-only classifier baseline — if plain text features transfer as well as activations, the interp claim weakens.
- [ ] **Confound direction:** check the deception direction isn't collinear with an obvious confound (refusal, sentiment, response length). Report cosine sim to a refusal/sentiment probe.

**Ablations (each is one axis, hold others fixed):**
- [ ] **Probe family:** diff-of-means vs logistic regression (vs LDA).
- [ ] **Layer:** full layer sweep transfer curves (does the best transfer layer differ from the best within-type layer?).
- [ ] **Token aggregation:** last vs mean vs last-response.
- [ ] **On-policy vs off-policy** data for the same type.
- [ ] **Training set size:** transfer vs n (data-efficiency curve).

**Optional but strong (ties to CAST/Lee et al., your steering interest):**
- [ ] **Causal validation via steering:** add/subtract the deception direction and measure change in deception rate on held-out prompts. A causal result elevates the paper from "we can classify" to "this direction is mechanistically meaningful."

**Done when:** every alternative explanation has a figure/table refuting it, and you know which ablation choices are robust vs fragile.

---

## Phase 6 — Robustness / stress tests

- [ ] **OOD scenarios:** eval each probe on held-out scenario families it never trained on (within its type). Transfer that survives OOD is far more convincing.
- [ ] **Prompt-format sensitivity:** re-run with a different chat template / system prompt; report AUROC delta.
- [ ] **Paraphrase robustness:** paraphrase eval prompts; probes should hold.
- [ ] **(If time) second model family** (e.g., a Llama-family model) as an external-validity check on ≥1 result.

**Done when:** you can state, with numbers, how much of the transfer signal survives distribution shift.

---

## Phase 7 — Reproducibility & release engineering

- [ ] One-command reproduction: `make results` (or a top-level script) that regenerates every figure from cached activations.
- [ ] Freeze `environment.yml`/`requirements.txt` with pinned versions (you've hit the NumPy 2.0 / torch and Py3.9 issues — pin explicitly).
- [ ] Update `CLAUDE.md` and `README` with: data provenance, how to regenerate, figure map, hardware used.
- [ ] Figure scripts as standalone, deterministic modules (matrix heatmap, layer curves, cosine-sim heatmap, scaling plot).
- [ ] **Data & model cards**; license check on any reused datasets; **ethics/dual-use statement** (deception detection is dual-use — address it briefly and seriously).
- [ ] Smoke tests + a CI check that the pipeline runs end-to-end on a tiny subset.

**Done when:** a stranger can clone, install, and regenerate Figure 1 from your instructions.

---

## Phase 8 — Write the paper

Write figures-first: the paper is really its figures + captions.

- [ ] **Freeze the figure set** (Fig 1 matrix, Fig 2 direction similarity, Fig 3 scaling, plus ablation tables). Write each caption to stand alone.
- [ ] **Outline** → Abstract, Intro (claim + contributions bullets), Related Work (Apollo/Goldowsky-Dill, LASR, Cundy & Gleave, Parrack, CAST — position precisely against each), Method, Datasets, Experiments, Results, Analysis/Discussion, Limitations, Ethics, Reproducibility.
- [ ] **Draft Method + Results first** (you have the material); Intro/Related last.
- [ ] **Limitations section** — write it honestly and early; reviewers respect it (small model scale, English-only, judge-labeled data, etc.).
- [ ] Internal review round with **Jack**, then **Dr. Zhu**; revise.
- [ ] Fill the venue's **reproducibility checklist**; build the appendix (dataset stats, full matrices per model, hyperparameters, extra layers).
- [ ] Post to **arXiv**, then submit to the chosen venue.

**Done when:** submitted, with the repo public and reproducible.

---

## Suggested immediate next 5 actions
1. Write the Phase 0 half-page (claim + definitions + hypotheses + venue) and get Dr. Zhu/Jack to sign off.
2. Lock the dataset schema (1a) and stand up the sycophancy set end-to-end as the pilot type.
3. Harden the small-model pipeline: seeds + CIs + logging (2c) on Qwen 0.5B.
4. Produce a **first 3×3 transfer matrix on Qwen 0.5B** with CIs — even rough. This validates the whole thesis cheaply before OSC spend.
5. Only then scale up the model ladder on OSC.

> Getting a scrappy full matrix on the smallest model *first* is the highest-leverage move — it tells you within a week whether the transfer story exists at all, before you invest in scaling, ablations, and writing.
