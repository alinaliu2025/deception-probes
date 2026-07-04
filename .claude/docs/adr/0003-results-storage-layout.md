# ADR 0003 — Results Storage Layout

**Status:** Accepted
**Date:** 2026-06-27

## Context

Experiments run on more than one machine (local Windows/CPU and OSC GPU nodes, see
ADR 0002) and by more than one person. The original code wrote every run to
fixed filenames under `results/` (e.g. `report_sycophancy_lr.png`), which has two
problems:

- **Clobbering.** A second run — same type, same method, different dataset size or
  a different machine — silently overwrites the first. There is no record that the
  earlier numbers ever existed.
- **No provenance.** A `.png` does not say which model, seed, dataset size, git
  commit, or host produced it. When two machines disagree, there is no way to tell
  what differed.

We also need a way to *share* numbers across machines without committing large
binary artifacts (`.npy` activations, `.png` figures) into git history.

## Decision

**ACCEPTED** — each run writes its own immutable, self-describing directory, and
the only git-tracked artifact is a small text `meta.json`.

| Decision | Choice | Rationale |
|---|---|---|
| Run directory | `results/runs/<utc>_<kind>_<method>_<sha>_<host>/` | Timestamp + commit sha + host make the name unique across runs *and* machines; nothing is overwritten. |
| Provenance file | `meta.json` per run | Plain text: full config (model, seed, dataset size, filter/permute flags) + metrics + library versions + git dirty flag. The self-describing record a `.png` can't be. |
| Source of truth | **The git-tracked `meta.json`** | `git pull` aggregates every machine's and collaborator's numbers; you diff JSON, not PNG pixels. |
| Heavy artifacts | `.npy`, `.npz`, `.png` stay **gitignored** | Regenerable from the metadata; committing them bloats history. |
| Implementation | `src/dprobe/runlog.py` (`new_run_dir`, `write_meta`) | Both scripts call it; provenance is captured in one place, not duplicated. |

The `.gitignore` idiom is "ignore the tree, re-include the directories at every
depth, then re-include the one file":

```gitignore
!results/runs/
results/runs/**
!results/runs/**/
!results/runs/**/meta.json
```

The `!results/runs/**/` line (re-including directories) is load-bearing: git will
not re-include a file if any parent directory is still excluded.

## Consequences

- New per-run artifacts go in the run dir, never back to a fixed top-level
  `results/` path. To track a new number, add a field to `meta.json` — don't invent
  a new tracked file type.
- To compare runs across machines: `git pull`, then read the `meta.json` files
  (each carries `auroc`, `best_layer`, `n_examples`, `model`, `git_sha`, host, …).
- A run is reproducible from its `meta.json` alone: re-run with the recorded model,
  seed, and flags to regenerate the gitignored figures/arrays.
- **Permutation control** (`--permute`, see README) writes a normal run dir with
  `kind="permctrl"` and `permuted: true` in its `meta.json`, so leak-check runs are
  archived alongside real ones and never confused with them.

## Open questions

- Whether to add a `scripts/runs.py` helper that tabulates every `meta.json` into
  one comparison table, or whether `git` + `jq` is enough for now.
- Whether to record the dataset content hash in `meta.json` so a result can be tied
  to the exact example set that produced it (matters once datasets grow and change).
