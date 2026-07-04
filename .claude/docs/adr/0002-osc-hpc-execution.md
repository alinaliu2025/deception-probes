# ADR 0002 — OSC HPC Execution for Larger Models

**Status:** Proposed
**Date:** 2026-06-27

## Context

The seed pipeline runs locally on `Qwen/Qwen2.5-0.5B-Instruct` (CPU/MPS friendly).
To get trustworthy AUROC we need to scale to larger instruct models, which need
real GPUs. The Ohio Supercomputer Center (OSC, project **PAS2324**) is the target.

Decisions needed: where models run, how big, how the model is switched without
git churn, where weights are cached, and how the Python env is built — while
keeping the existing local workflow untouched for fast iteration/testing.

Inputs informing the sizing:
- Sycophancy LR probe on 0.5B, 5000 examples → **AUROC 0.84**. Signal is already
  present at the smallest scale, so larger models should sharpen the directions,
  not rescue a null result. Low risk in scaling up.
- Collaborator (grad student) guidance: validate code on **V100** first, run
  **formal experiments on A100**, treat **H100 as opportunistic** if the queue is
  acceptable. For model size, **1B–10B is the working range**; Qwen ~9B is "large
  enough to capture most concepts"; most current mech-interp papers use <10B.
  Include **one or two models above 10B** for the final paper if time/resources allow.

OSC facts that constrain the design:
- Three GPU clusters: **Cardinal** (4× H100 94GB/node), **Ascend** (A100 40/80GB,
  AMD EPYC), **Pitzer** (V100 32GB).
- Jobs are SLURM `sbatch`; every job needs `#SBATCH --account=PAS2324`.
- OSC prefers `--gpus-per-node=N` over `--gres=gpu:N`; default `--gpu_cmode=shared`.
- `$HOME` has a small quota and throttled I/O on compute nodes; bulk data belongs
  on `/fs/scratch/<proj>` (fast, ~90-day purge) or `/fs/ess/<proj>` (persistent).
- Python: `module load miniconda3`, build a prefixed env, activate with
  `source activate` — **not** `conda activate` (it rewrites shell rc files).
  Set `PYTHONNOUSERSITE=True` to isolate from `~/.local`.

## Decision

**PROPOSED** — add a thin OSC execution path; do not change the science code's behaviour.

| Decision | Choice | Rationale |
|---|---|---|
| Primary model | **Qwen2.5-7B-Instruct** | Centre of the 1–10B working range; single-GPU on V100-32GB and A100; existing `.to(device)` code stays valid. |
| Cluster workflow | **V100 smoke → A100 formal → H100 opportunistic** | Collaborator's path. V100 (compute 7.0 → fp16) just confirms the run; A100-80GB runs the real 7B sweeps; H100/Cardinal only if queue is short. |
| Model ladder | 0.5B (done, AUROC 0.84) → 1.5/3B sanity → **7B formal** → 14B paper stretch | Sweep, not a single baseline. "1–2 models >10B" (14B) reserved for the final paper if time allows. |
| Model ceiling (this phase) | **≤14B, single GPU** | Keeps `model.to(device)` valid; no `device_map="auto"` rework. 32B+ deferred. |
| Model switching | **`DPROBE_MODEL` env var** in `config.py` (`os.environ.get`, default 0.5B) | One sbatch sweeps sizes via `--export`; no per-run edits to a tracked file. |
| Weight cache | **`HF_HOME=/fs/scratch/PAS2324/hf`**, set in sbatch | Keeps 15–30GB downloads off `$HOME`. Re-download after purge is cheap. |
| Python env | **conda prefix env** `/fs/ess/PAS2324/dprobe-env`, `source activate`, `PYTHONNOUSERSITE=True` | OSC-recommended; persistent so not purged; `pip install -e ".[dev]"` into it. |
| Local workflow | **Unchanged** | Default model stays 0.5B; `get_device()` already falls back to cpu/mps. OSC is opt-in via env + sbatch only. |

## Planned artifacts

1. `src/dprobe/config.py:5` — `MODEL_NAME = os.environ.get("DPROBE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")` (backward-compatible; local default unchanged).
2. `scripts/osc/compare.sbatch` — **Ascend, 1× A100-80GB** (`--account=PAS2324`, `--gpus-per-node=1`), `HF_HOME` on scratch, conda activate, default `DPROBE_MODEL=Qwen/Qwen2.5-7B-Instruct`, runs `python -m scripts.compare --method lr`. Override per submit: `sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct ...`. A V100 variant (or `--cluster` swap) is used only for the initial smoke run.
3. `docs/HPC.md` — one-time env build steps + submit/monitor/cancel cheatsheet (`sbatch`, `squeue -u $USER`, `scancel`, `sinteractive` for debugging) + the V100→A100→H100 progression.

## Consequences

- `_pick_dtype` (activations.py:26) already handles every target: V100 (7.0 → fp16), A100/H100 (≥8.0 → bf16). No change. 7B in fp16/bf16 ≈ 14GB → comfortable on A100-80GB.
- Outputs (`results/`) stay gitignored; sbatch writes `results/slurm-%j.out`.
- Re-running after a scratch purge re-downloads weights — acceptable, not silent data loss.
- **Out of scope (future ADR):** multi-GPU sharding for 32B+ (`device_map="auto"`, `.to(device)` rework), Cardinal/H100 as the formal target, OnDemand/Jupyter interactive use.

## Open questions

- Walltime ceiling per run at 7–14B — start with `--time=2:00:00`, tune after first job.
- Whether to pin model revisions for reproducibility once a model is chosen for real results.
- **Methodology gate (blocking):** Phase 1 is locked to Qwen2.5-0.5B-Instruct (memory `project_phase1.md`, CLAUDE.md). Moving the formal baseline to 7B is a real methodology change — confirm with collaborators (Zhu, Zhu, Liu) before flipping. ADR stays **Proposed** until then.
