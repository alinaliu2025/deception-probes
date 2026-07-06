# CLAUDE.md

Persistent context for Claude Code. Loaded at the start of every session. Keep it
short and factual: project layout, conventions, commands, "always do X" rules.

## What this project is

Research code for the lab (PI: Dr. Zhu). We train linear **deception probes** on the
residual stream of a small local LLM and compare three single-turn deception types:
**sycophancy, sandbagging, omission/concealment**. The headline result is a
cross-type **transfer matrix** (does a probe for one type detect another?).

## Layout

- `src/dprobe/` installable package (`pip install -e .`)
  - `activations.py` load model, extract last-token residual stream per layer
  - `probes.py` `Probe` object + `fit_mms` / `fit_lr`
  - `evaluate.py` layer sweep, AUROC, `transfer_matrix`, `direction_cosines`
  - `plotting.py` per-type report + comparison heatmaps
  - `runlog.py` per-run output dirs (`new_run_dir`, `write_meta`)
  - `data/` one module per deception type, each exposes `build() -> list[Example]`
- `scripts/train_one.py` train+report a single type
- `scripts/compare.py` the cross-type study
- `tests/test_smoke.py` fast math/dataset checks, no model download

## Commands

- Install: `pip install -e ".[dev]"`
- Fast tests: `pytest -q`
- One type: `python -m scripts.train_one --type sycophancy`
  - `--max-examples N` cap dataset (drops whole pairs, keeps balance)
  - `--permute` permutation control: shuffle labels, expect AUROC ≈ 0.5 (else leak)
  - `--filter` behaviour filters: sandbagging capability filter (see IMPORTANT);
    sycophancy `--design framing` keep-if-flips; sycophancy `--design behavioral`
    assigns the labels (mandatory there)
  - `--design {completion,framing,behavioral,rollout}` sycophancy constructions
    (ADR 0006/0007/0008); completion and framing leak, behavioral is
    content-confounded (neutral-read control hit AUROC 1.0, ADR 0008), rollout
    is the within-question fix (Proposed, pending team sign-off)
  - `--read-prompt neutral` behavioral-design confound control;
    `--rollouts/--temperature/--max-new-tokens` rollout-design sampling knobs;
    `--rollout-prefix {commit,text}` read-prefix mode (text = wording-shortcut
    ablation, ADR 0009 addendum)
  - `--source {factual,factual-small}` sycophancy behavioral/rollout only:
    `factual` = ARC MCQs, user asserts a wrong answer (ADR 0009, Proposed);
    `factual-small` = tiny repo-resident OFFLINE fixture
    (`src/dprobe/data/fixtures/factual_smoke.jsonl`) for smoke tests/demos, no
    network, NOT trustworthy AUROC
  - `--gate {logprob,sampled}` sycophancy behavioral/rollout belief gate:
    `logprob` (default) = one deterministic teacher-forced comparison (ADR 0007);
    `sampled` = generate `--gate-n` (20) answers unpressured, keep if correct on
    ≥ `--gate-threshold` (0.9) at `--gate-temperature` (0.7) — the "model is SURE"
    gate; opt-in, costs a generate call per question
  - rollout runs write `results/runs/<run>/run_log.txt`: per-question gate result,
    parsed rollout outcomes, and set membership (USED / AMBIVALENT-BUT-TRIMMED /
    SINGLE-CLASS). See `.claude/docs/rollouts-explained.md` for a plain-language
    walkthrough of the rollout design.
- Full comparison: `python -m scripts.compare --method lr`

## ADRs

Architecture Decision Records live in `.claude/docs/adr/`. Use sequential numbering (`0001-`, `0002-`, …).

## Conventions

- Label **1 = deceptive condition, 0 = control**. Never flip this.
- A matched control/deceptive pair must differ ONLY in what induces the behaviour
  (system prompt or user framing), never the underlying task.
- New deception type = new module in `src/dprobe/data/` + one line in
  `data/__init__.py`. Don't special-case types elsewhere.
- Probes are layer-specific; a `Probe` carries its own `.layer`.
- Each run writes its own dir `results/runs/<utc>_<kind>_<method>_<sha>_<host>/`
  (via `runlog`). Only `meta.json` is git-tracked (cross-machine source of truth);
  heavy `.npy`/`.png` artifacts are gitignored and regenerable.

## IMPORTANT

- Sandbagging needs the **capability filter** or results are meaningless. Only count a
  wrong answer as sandbagging if the model answers correctly under the control prompt.
- The seed datasets are tiny (12 items). They exist to make the pipeline run, NOT to
  produce trustworthy AUROC. Scale datasets before drawing conclusions.
