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
  - `--filter` sandbagging only (capability filter, see IMPORTANT)
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
