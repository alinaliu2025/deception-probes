#!/bin/bash
# Steer beta-sweep (ablate pass), Qwen2.5-7B, Pitzer V100-32GB (fp16).
#
# Pushes the ablation coefficient beta far past the default top (4) to test
# whether the caving direction is causally NECESSARY: does projecting more of it
# out (overcorrection) flip more of the baseline-caved items back to correct?
# The 2026-07-21T19-42-20Z run showed only ~3/58 items flip by beta=4 with
# parse_rate pinned at 1.0 -- i.e. no generation breakdown, so beta has headroom.
#
#   sbatch scripts/osc/steer_beta_qwen.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# V100 (compute 7.0, fp16) MUST match the card the probe's model was run on:
# a direction is model+dtype-specific. The probe.npz below was trained fp16 on a
# V100 (p0301); _pick_dtype gives fp16 on V100 / bf16 on A100, so a V100 keeps
# the activations v was learned on. meta.json records model_dtype -- confirm fp16.
# 48core = the 2x V100-32GB node (7B fp16 ~14GB fits one GPU); 40core OOMs.
#SBATCH --job-name=steer-beta-qwen7b
#SBATCH --account=PAS2324
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48core
#SBATCH --time=02:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime 2h: steering has NO belief gate (the train_one bottleneck). It loads
# the 7B once, generates a 778-item baseline, then sweeps 9 betas x 2 directions
# (probe + random control) over the ~58 baseline-caved items, greedy. Cheap.

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: needs the --betas flag (ablate beta-sweep). Before submitting, on a
# login node in $REPO: `git fetch && git checkout sycophancy_v2 && git pull`.
# meta.json records git_sha -- confirm it matches the commit that adds --betas.
#
# PROBE FILE: results/runs/.../probe_promptfinal.npz is gitignored -- it must
# already be present in $REPO (it is, if the training run wrote it here on OSC).

export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1

# confirm the card BEFORE loading 14.6GB of weights -- 7B fp16 OOMs on a 16GB GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m scripts.steer --probe results/runs/2026-07-21T14-01-53Z_sycophancy_mms_8197a67_p0301/probe_promptfinal.npz --model Qwen/Qwen2.5-7B-Instruct --source factual --split test --mode ablate --control random --forced-choice --max-new-tokens 16 --betas 0,1,2,4,6,8,12,16,24
