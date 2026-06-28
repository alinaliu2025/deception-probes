# OSC HPC notes (project PAS2324)

Reference for running the deception-probes pipeline on the Ohio Supercomputer
Center. Decisions/rationale live in [`.claude/docs/adr/0002-osc-hpc-execution.md`](../.claude/docs/adr/0002-osc-hpc-execution.md);
this file is the operational cheatsheet.

## Two ways to run — both go through SLURM

OSC OnDemand's **Interactive Desktop** sessions are still SLURM jobs; OnDemand
writes and submits the sbatch for you. You always get a queued, walltime-limited
allocation. The only difference is whether *you* write the script.

| Path | Use for | How |
|---|---|---|
| **Interactive Desktop** (OnDemand → My Interactive Sessions) | dev, debugging, smoke tests | Pick cluster + **a GPU node type** (see below), wait for allocation, then use the desktop or `ssh` into the node and run code. |
| **`sbatch`** (batch script) | formal experiment sweeps | Survives disconnect, no idle-GPU waste, queue many runs. `sbatch scripts/osc/compare.sbatch`. |

**Workflow:** V100 smoke (interactive) → A100 formal (batch) → H100 opportunistic.

> ⚠️ Must select a **GPU-enabled** node type. "Standard Compute" / "any" nodes
> have **no GPU** — the code falls back to CPU.

## Node catalog (OnDemand Desktop options)

### Pitzer — V100 (smoke tests)
- **Standard Compute** — no GPU. 224×40-core + 340×48-core, all 192 GB RAM.
- **GPU Enabled** — NVIDIA Tesla V100:
  - 40-core node → **2× V100 16 GB**
  - 48-core node → **2× V100 32 GB**  ← pick this; 7B fp16 (~14 GB) fits one GPU
  - densegpu → 4× V100 16 GB
  - Visualization nodes = GPU + X server (VirtualGL); not needed here.
- **Large Memory** — largemem 48-core/768 GB; hugemem 80-core/3 TB. No GPU relevance.

### Ascend — A100 (formal experiments)
- **any** (88 cores) — fastest queue, but **no guaranteed GPU**.
- **vis** (88 cores) — 1× A100 + X server (VirtualGL). For 3D viz only.
- **gpu** (88 cores) — **1× A100, CUDA, no X server**  ← formal-run target. ~96 such nodes.

### Cardinal — H100 (opportunistic)
- **Standard Compute** — no GPU. 326×96-core, 512 GB RAM.
- **GPU Enabled** — NVIDIA H100. 96 cores, 1 TB RAM.
- **Visualization Nodes** — GPU + X server (VirtualGL).
- **Huge Memory** — 2 TB RAM, 16 nodes. No GPU need here.

## One-time environment setup (on a login node)

```bash
module load miniconda3
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --prefix /fs/ess/PAS2324/dprobe-env python=3.10
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env   # NOT `conda activate`
pip install -e ".[dev]"
```

## Caches off $HOME (small quota + slow node I/O)

```bash
export HF_HOME=/fs/scratch/PAS2324/hf   # scratch: fast, big, ~90-day purge
```

## Switching model size (no code edits)

`config.py` default stays 0.5B for local runs. Override via env var:

```bash
export DPROBE_MODEL=Qwen/Qwen2.5-7B-Instruct
python -m scripts.compare --method lr
# or, in batch:  sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/compare.sbatch
```

## Job cheatsheet (batch path)

```bash
sbatch scripts/osc/compare.sbatch   # submit
squeue -u $USER                     # my queue
scancel <jobid>                     # cancel
OSCfinger $USER                     # confirm SLURM accounts (PAS2324)
sinteractive -A PAS2324 -g 1        # quick interactive GPU shell
```
```
