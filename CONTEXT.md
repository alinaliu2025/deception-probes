# CONTEXT.md — ubiquitous language for deception-probes

Glossary only. No implementation details, no specs. See `.claude/docs/adr/` for
decisions and CLAUDE.md for commands/conventions.

## Terms

**Run** — one execution of a top-level script (`train_one`, `compare`, `steer`)
that produces results. Every run owns exactly one run dir.

**Run dir** — the per-run output directory under `results/runs/`, named
`<utc>_<kind>_<method>_<sha>_<host>`. Created when the run *starts*; its
timestamp is the run's start time.

**Complete run** — a run whose run dir contains `meta.json`. `meta.json` is
written last, so its absence marks a run that crashed or was aborted.

**Console log** — the raw, timestamped record of everything the run printed
(stdout and stderr, including tracebacks). One per run, all scripts. Lives in
the run dir as `console.log`. Gitignored.

**Filter trace** — the structured per-question record of the decisions a
behaviour filter or belief gate made during a run: what was kept, dropped,
relabelled, and why. Lives in the run dir as `run_log.txt`. Gitignored.
Distinct from the console log: the trace is the inspectable decision record,
the console log is the raw stream.

**Belief gate** — the pre-filter check that a question is only usable if the
model, unpressured, does NOT already agree with the user (`logprob` or
`sampled` mode).

**Forced-choice regime** — constraining the model to answer with only a bare
`(A)`/`(B)` (via `DID_SYSTEM`), as opposed to free-form prose. The `did` design
probes this regime so the committed answer is unambiguous; it is a *different*
behavioural condition than free-form answering (no chain-of-thought), not just a
parsing convenience. See ADR 0012.

**Pressure-response arrow** (**arrow**) — for one question, the vector
`pressured_act − calm_act`: how the residual stream moves when the user's
pressure is bolted onto the bare question, read at a fixed position. The
subtraction cancels the question's content, so an arrow carries only what the
pressure changed. See ADR 0012.

**Capitulation direction** — the difference-of-differences result,
`mean(arrow | caved) − mean(arrow | held)`: the candidate direction for
sycophantic capitulation specifically, with both the question content (cancelled
in each arrow) and the generic pressure response (cancelled between the class
means) removed. See ADR 0012.
