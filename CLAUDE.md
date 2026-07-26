# CLAUDE.md

Persistent context for Claude Code. Loaded at the start of every session. Keep it
short and factual: project layout, conventions, commands, "always do X" rules.

## ALWAYS DO FIRST: read `TODO.md` and `docs/SCOPE.md`

Every session, before anything else. `TODO.md` has the single next action at the top.
`SCOPE.md` says what is in scope and what is explicitly deferred until 2026-09-06.

As of 2026-07-26 the project is scoped to **one thing**: in-context scheming on a
reasoning model (Qwen3-8B), branch `scheming`. Sycophancy, sandbagging, omission,
eval-awareness, the 3x3 transfer matrix, and the model ladder are all **out of scope**.
Most of what's below this line describes that deferred work; it is history, not the
current plan.

New ideas go in the parking lot at the bottom of `SCOPE.md` with a date. Not into the
branch, and not into a new design.

## Writing voice (read before drafting prose)

When writing anything for Alina (chat, ADRs, docs, comments), follow
`.claude/docs/writing/anti-ai-writing-style.md` and
`.claude/docs/writing/voice_profile_alina_liu.md`. Short version: no em dashes, no
"it's not X, it's Y" reframes, no padded three-item lists, digits not words,
contractions, plain punctuation, take a stance, stop when the point lands.

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
  - `--design {completion,framing,behavioral,rollout,did}` sycophancy
    constructions (ADR 0006/0007/0008/0012); completion and framing leak,
    behavioral is content-confounded (neutral-read control hit AUROC 1.0,
    ADR 0008), rollout is the within-question fix (Proposed), did is the
    difference-of-differences fix — pressured-minus-calm ARROW cancels question
    content, then diff-of-means on arrows is the capitulation direction; writes a
    probe+report for BOTH read positions (promptfinal = clean, answertoken =
    diagnostic, their AUROC gap = the letter shortcut), needs `--filter`, gate
    defaults to `sampled`; probes the FORCED-CHOICE regime (`DID_SYSTEM` makes the
    model answer bare `(A)/(B)` — a different behavioural condition than free-form,
    chosen after free-form answers mislabelled 41/61 caves via negation-led
    rebuttals) (ADR 0012, Proposed, pending team sign-off)
  - sandbagging did (ADR 0012 port): `--type sandbagging --design did --source
    factual --filter [--pressure {instructed,incentive}]` — same DiD mechanics,
    but the pressure lives in the SYSTEM prompt, so `meta['calm_system']`
    (CONTROL + forced-choice suffix) routes the belief gate and the calm arrow
    side; user text is identical calm/pressured. `--pressure incentive` =
    evaluation-incentive-only prompt (ADR 0011 addendum).
  - `--read-prompt neutral` behavioral-design confound control;
    `--rollouts/--temperature/--max-new-tokens` rollout-design sampling knobs;
    `--rollout-prefix {commit,text}` read-prefix mode (text = wording-shortcut
    ablation, ADR 0009 addendum)
  - `--source {factual,factual-small}` sycophancy behavioral/rollout/did only:
    `factual` = ARC MCQs, user asserts a wrong answer (ADR 0009, Proposed);
    `factual-small` = tiny repo-resident OFFLINE fixture
    (`src/dprobe/data/fixtures/factual_smoke.jsonl`) for smoke tests/demos, no
    network, NOT trustworthy AUROC
  - `--gate {logprob,sampled}` sycophancy behavioral/rollout/did belief gate
    (did defaults to `sampled`, the consistency prerequisite; others `logprob`):
    `logprob` = one deterministic teacher-forced comparison (ADR 0007);
    `sampled` = generate `--gate-n` (20) answers unpressured, keep if correct on
    ≥ `--gate-threshold` (0.9) at `--gate-temperature` (0.7) — the "model is SURE"
    gate; opt-in, costs a generate call per question
  - every run writes `console.log` into its run dir (timestamped tee of ALL
    stdout/stderr incl. tracebacks) and every per-item decider (all filters, the
    belief gate, steering) writes `run_log.txt` — the filter trace: gate
    evidence, full generated texts, keep/drop/relabel verdicts (ADR 0011).
    Rollout traces show set membership (USED / AMBIVALENT-BUT-TRIMMED /
    SINGLE-CLASS); see `.claude/docs/rollouts-explained.md` for a plain-language
    walkthrough of the rollout design.
- Full comparison: `python -m scripts.compare --method lr`
- Steering (causal check, ADR 0010): `python -m scripts.steer --probe
  results/runs/<run>/probe.npz --model Qwen/Qwen2.5-7B-Instruct --source factual`
  - Validates that a probe DIRECTION causes caving, not just correlates. `add` pass
    adds `alpha·v` on unpressured items (wrong-rate should climb with alpha);
    `ablate` pass projects `v` out on pressured items (caving should drop).
  - `--probe` needs the run's `probe.npz` (gitignored; scp it back from OSC).
    `--source factual-small` runs offline for a plumbing check (not trustworthy).
  - `--alphas` are multiples of the layer's residual norm (`--raw` = absolute);
    `--mode {add,ablate,both}`, `--samples N` (1=greedy), `--split {train,test}`.

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
- The run dir is created at the START of a run; `meta.json` is written last, so a
  dir without it is a crashed/aborted run and its `console.log` is the post-mortem
  (ADR 0011). Vocabulary: `console.log` = raw timestamped stream, `run_log.txt` =
  structured filter trace (see CONTEXT.md).
- A new filter should set `last_stats` (goes into meta.json) AND `last_log` (the
  filter trace, written to run_log.txt); render traces with `dprobe.tracefmt`.

## IMPORTANT

- Sandbagging needs the **capability filter** or results are meaningless. Only count a
  wrong answer as sandbagging if the model answers correctly under the control prompt.
- The seed datasets are tiny (12 items). They exist to make the pipeline run, NOT to
  produce trustworthy AUROC. Scale datasets before drawing conclusions.
