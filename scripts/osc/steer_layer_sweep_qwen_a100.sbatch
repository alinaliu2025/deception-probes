#!/bin/bash
# Causal steering-LAYER sweep, Qwen2.5-7B, Ascend A100-80GB (bf16).
# V100/Pitzer variant: steer_layer_sweep_qwen_v100.sbatch (slower, easier to get).
#
# Sweeps candidate STEERING layers in an asymmetric band around the detection
# peak L* (from the 14-01-53Z DiD run, best_layer 24) and picks the layer whose
# DiD capitulation direction most strongly DRIVES caving while staying coherent.
# This is layer selection for intervention, NOT a detection/AUROC sweep and NOT a
# probe train -- it rebuilds a fresh diff-in-means arrow direction at every band
# layer (the saved probe.npz holds L24 only), so it needs no probe file, just L*.
#
#   sbatch scripts/osc/steer_layer_sweep_qwen_a100.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=ascend <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# A100 (Ascend, compute 8.0) is chosen for SPEED -- _pick_dtype gives bf16 here
# (fp16 on a V100), and the 80GB card fits 7B with room to spare (no memory
# constraint needed, unlike the 32GB-pinned V100 script). This script LOADS NO
# saved probe: it builds AND steers the direction in the same run on this card, so
# there is no cross-dtype hazard (the steer.py cross-dtype caveat does not apply).
# L*=24 came from a V100 fp16 run; it is only the band CENTRE (band is L*-8..L*+3),
# so a small bf16 shift in the exact AUROC peak is absorbed. meta.json records
# model_dtype. Tradeoff vs the V100 script: faster card, but the quad partition is
# often deep-queued -- check `squeue -u $USER --clusters=ascend --start`.
#SBATCH --job-name=steerlayer-qwen7b-a100
#SBATCH --account=PAS2324
#SBATCH --clusters=ascend
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
# Ascend caps CPUs per job to a share of the node proportional to the GPUs asked
# for (4 A100 + ~88 cores/node -> ~22 cores/GPU). Without an explicit request the
# job inherits more than that share and Slurm rejects it with "More processors
# requested than permitted". 12 cores is plenty for single-GPU inference (CPU only
# feeds tokenisation/dataloading) and safely under the per-GPU cap.
#SBATCH --time=03:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime 3h (A100 is ~2-3x faster than the V100, so a tighter margin suffices):
# two costs stack. (1) Labeling the full dual-ordering factual set (~14.8k
# questions) with the sampled gate (gate-n generate calls each) is the same
# bottleneck as did_qwen.sbatch. (2) The sweep is 12 layers x 19 alphas, and the
# cave pass generates over every baseline-hold test item (~761) at each point. If a
# real run finishes well under 3h, trim this. To cut cost: cap the labeling with
# `--max-train N`, cap the eval set with `--max-items N`, or drop the always-broken
# alpha tail (2.5,3,4).

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: needs the steer_layer_sweep script (and the steer.py-matched --alphas
# default). Before submitting, on a login node in $REPO:
# `git fetch && git checkout sycophancy_v2 && git pull`. meta.json records git_sha.

# per-user cache: a shared HF_HOME fails with PermissionError the first time
# someone pulls a model its owner didn't
export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1   # keeps 14GB of download bars out of the log

# confirm the card BEFORE loading 14.6GB of weights
nvidia-smi --query-gpu=name,memory.total --format=csv

# L*=24 is the DiD detection peak (14-01-53Z run). --alphas is the steer.py add-pass
# distribution (dense sub-1.25, where coherence holds); it is the committed default,
# passed here so the log is self-documenting.
python -m scripts.steer_layer_sweep --best-layer 24 --direction did --source factual --gate sampled --gate-n 10 --max-new-tokens 16 --model Qwen/Qwen2.5-7B-Instruct --alphas 0,0.25,0.5,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.5,2,2.5,3,4
