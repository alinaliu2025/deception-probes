#!/bin/bash
# Causal steering-LAYER sweep, Qwen2.5-7B, Pitzer V100-32GB (fp16).
# A100/Ascend variant: steer_layer_sweep_qwen_a100.sbatch (faster, deeper queue).
#
# Sweeps candidate STEERING layers in an asymmetric band around the detection
# peak L* (from the 14-01-53Z DiD run, best_layer 24) and picks the layer whose
# DiD capitulation direction most strongly DRIVES caving while staying coherent.
# This is layer selection for intervention, NOT a detection/AUROC sweep and NOT a
# probe train -- it rebuilds a fresh diff-in-means arrow direction at every band
# layer (the saved probe.npz holds L24 only), so it needs no probe file, just L*.
#
#   sbatch scripts/osc/steer_layer_sweep_qwen_v100.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# Pitzer V100-32GB (compute 7.0) is chosen for AVAILABILITY -- the Ascend A100 quad
# partition is deep-queued. _pick_dtype gives fp16 on a V100. This script LOADS NO
# saved probe: it builds AND steers the direction in the same run on this card, so
# there is no cross-dtype hazard (the steer.py cross-dtype caveat does not apply);
# fp16 here also MATCHES the fp16 V100 run L*=24 came from. meta.json records
# model_dtype.
#
# 32GB IS REQUIRED: 7B fp16 weights are ~14GB, which will OOM a 16GB V100 once the
# generation KV cache is added. The 32GB V100s are a SUBSET of the gpu-partition
# nodes, so the job MUST pin them with a feature constraint (below). NOTE:
# --constraint=48core ALONE failed with "Requested node configuration is not
# available" (the gpu partition does not schedule a single GPU on those nodes that
# way). Discover the right feature/GRES for the 32GB cards first:
#   sinfo -M pitzer -p gpu -N -o "%N %f %G"
# then set the constraint on the matching feature. The line below is the common OSC
# label; if it errors the same way, swap it for whatever %f tags the 32GB rows.
#SBATCH --job-name=steerlayer-qwen7b-v100
#SBATCH --account=PAS2324
#SBATCH --clusters=pitzer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100-32g
#SBATCH --time=06:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime 6h (V100 is ~2-3x slower than the A100, so a wider margin than an Ascend
# run): two costs stack. (1) Labeling the full dual-ordering factual set (~14.8k
# questions) with the sampled gate (gate-n generate calls each) is the same
# bottleneck as did_qwen.sbatch. (2) The sweep is 12 layers x 19 alphas, and the
# cave pass generates over every baseline-hold test item (~761) at each point. The
# 14-00-13Z run did all this on a V100S-32GB inside its walltime; if a real run
# finishes well under 6h, trim this. To cut cost: cap the labeling with
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
