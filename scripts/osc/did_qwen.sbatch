#!/bin/bash
# DiD sycophancy run, Qwen2.5-7B, Pitzer V100-32GB (fp16).
#
# Reproduces the 2026-07-15 run 2026-07-15T18-43-28Z_..._368d4b6_p0302
# (promptfinal AUROC 0.934) but with the ADR 0012 letter fixes now in the code:
# BOTH letter orderings per factual question + asserted-letter balance in
# did_filter. Nothing in the command changed -- the fixes are automatic in the
# `factual` source and did_filter -- so any AUROC move vs 0.934 is the fix, not a
# knob change.
#
#   sbatch scripts/osc/did_qwen.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# V100 (compute 7.0) is deliberate: the 0.934 run was fp16 on p0302 (a V100), and
# _pick_dtype gives fp16 on V100 / bf16 on A100. Matching the card matches the
# dtype so the comparison is apples-to-apples; meta.json records `model_dtype`.
# 48core = the 2x V100-32GB node (7B fp16 ~14GB fits one GPU); 40core is 16GB and
# will OOM.
#SBATCH --job-name=did-qwen7b
#SBATCH --account=PAS2324
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48core
#SBATCH --time=08:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime is 8h, not the 3h the OLMo sbatch uses: dual ordering (ADR 0012)
# DOUBLES the pre-gate question count (7398 -> ~14.8k), and the sampled belief
# gate (gate-n generate calls per question) is the run's bottleneck, so the gate
# stage roughly doubles. Trim later once a real run reports its wall time.

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: this run needs the ADR 0012 letter-fix commit. Before submitting,
# on a login node in $REPO: `git fetch && git checkout sycophancy_v2 && git pull`
# (or the branch/sha you pushed). meta.json records git_sha -- confirm it matches.

# per-user cache: a shared HF_HOME fails with PermissionError the first time
# someone pulls a model its owner didn't
export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1   # keeps 14GB of download bars out of the log

# confirm the card BEFORE loading 14.6GB of weights -- 7B fp16 OOMs on a 16GB GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m scripts.train_one --type sycophancy --design did --source factual --filter --method mms --gate-n 10 --max-new-tokens 16 --model Qwen/Qwen2.5-7B-Instruct
