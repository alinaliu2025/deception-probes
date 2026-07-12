# ADR 0011 — Full Run Logging (console.log + universal filter traces)

**Status:** Accepted
**Date:** 2026-07-11

## Context

Before this change, the only record of *what happened during* a run was its
stdout — lost when the terminal scrolls, or stranded in a SLURM `.out` file
divorced from the run dir it belongs to. The structured per-question trace
(`run_log.txt`) existed only for the sycophancy **rollout** filter; the
behavioral, framing, and sandbagging filters, the sampled belief gate, and the
steering passes all made per-item keep/drop/relabel decisions invisibly.

Worse, `new_run_dir()` was called at the **end** of each script, after model
load, filtering, extraction, and the layer sweep — so there was nowhere to
write a log *during* the interesting parts, and a crashed run left no run dir
at all. Crash logs are precisely the ones worth keeping.

Terminology was also muddled: "run log" was used both for the rollout trace
file and informally for "the record of the run". CONTEXT.md now pins the
vocabulary: **console log** = raw timestamped stream; **filter trace** =
structured per-question decision record.

## Decision

**ACCEPTED** — every run records everything, in two artifacts per run dir.

| Decision | Choice | Rationale |
|---|---|---|
| Run dir lifecycle | Created at the **start** of every script; `meta.json` written last | A dir **without `meta.json` is a crashed/aborted run** and its `console.log` is the post-mortem. Dir-name timestamp is now the run's *start* time. |
| Console capture | `runlog.capture_console(run_dir)` tees stdout+stderr into `console.log` | Terminal/SLURM output is byte-identical; the file gets UTC-timestamped lines (free per-stage durations), `\r` progress counters collapsed to their final frame, and any traceback written before the exception propagates. |
| Filter traces | **Every per-item decider** writes `run_log.txt` via its `last_log` attribute | rollout (existing, now with gate evidence + full rollout texts), behavioral, framing keep-if-flips, sandbagging capability filter, and a per-item steering trace (`render_steer_log`). `compare.py` makes no per-item decisions → console.log only. |
| Trace content | Human-readable text, **full generated texts inline** (JSON-quoted one-liners) | Both files are gitignored (only `meta.json` is tracked, ADR 0003), so size is not a git concern; full texts are exactly what debugging unparsed rollouts / broken steering needs. |
| Naming | `console.log` (new) + `run_log.txt` (kept) | Renaming `run_log.txt` would orphan old run dirs and existing docs; the vocabulary fix lives in CONTEXT.md instead. |
| Shared format | `src/dprobe/tracefmt.py` (`short`, `qtext`, `render`) | All traces read alike; renderers stay next to their filters. |
| Gate evidence | Belief gates return `(passed, info)` | `logprob` gate records both sequence log-probs; `sampled` gate records every sampled answer text + its parse. The trace shows *why* a question was gated, not just that it was. |

## Consequences

- `results/runs/` will accumulate dirs for crashed/aborted runs. That is a
  feature (the log survives), but sweeps of incomplete dirs (`meta.json`
  absent) are now meaningful cleanup targets.
- `.gitignore` needs no change: `console.log` and `run_log.txt` fall under the
  existing "everything but `meta.json`" rule.
- Filters expose `last_stats` (→ `meta.json`) **and** `last_log`
  (→ `run_log.txt`); a new deception type's filter should set both.
- `scripts/*` now split argument validation (before the run dir exists —
  usage errors don't create dirs) from the run body (inside
  `capture_console`).
- The belief-gate helpers changed signature (`bool` → `(bool, dict)`); any
  new caller must unpack the tuple.

## Open questions

- Whether a machine-readable `trace.jsonl` beside the human-readable
  `run_log.txt` earns its keep once someone wants to *analyse* rollout texts
  rather than read them (deferred: YAGNI until a consumer exists).
