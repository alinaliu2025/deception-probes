#!/bin/bash
# STEP 2 of 2 -- scaled ablation (beta-sweep) at the STEERING layer L20,
# Qwen2.5-7B, Pitzer V100-32GB (fp16). Needs step 1
# (dump_l20_direction_qwen_v100.sbatch) to have produced results/probes/probe_L20.npz.
#
# Tests whether the DiD capitulation direction is causally NECESSARY *at the layer
# where it actually controls behaviour*. The prior beta-sweep (2026-07-21) ablated
# the L24 DETECTION probe and was a null (~1/58 items flip) -- but the layer sweep
# showed cave-induction collapses above L20, so L24 was the wrong layer to ablate.
# This ablates the L20 direction instead: h' = h - beta*(h.v)v on the baseline-CAVED
# test items, beta=1 removes the whole component, beta>1 overcorrects. A --control
# random arm ablates a matched random unit vector over the SAME items so a drop is
# attributable to the DIRECTION, not generic perturbation.
#
#   sbatch scripts/osc/steer_ablate_l20_qwen_v100.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# Chain after step 1 without babysitting:
#   jid=$(sbatch --parsable scripts/osc/dump_l20_direction_qwen_v100.sbatch)
#   sbatch --clusters=pitzer --dependency=afterok:$jid scripts/osc/steer_ablate_l20_qwen_v100.sbatch
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# WHY V100 fp16: MUST match step 1's card/dtype -- probe_L20.npz was built there,
# and a direction is model+dtype-specific (steer.py cross-dtype caveat). fp16 also
# matches the run L20 was recommended from (23-19-47Z, p0331). meta.json records
# model_dtype -- confirm fp16. 32GB required (7B fp16 ~14GB + KV cache OOMs 16GB).
#SBATCH --job-name=ablate-l20-qwen7b-v100
#SBATCH --account=PAS2324
#SBATCH --clusters=pitzer
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime 2h: steering has NO belief gate (the labeling bottleneck lives in step 1).
# It loads the 7B once, generates a ~778-item baseline to find the baseline-caved
# set (~58 on the clean test split), then sweeps 9 betas x 2 directions (probe +
# random control) over those items, greedy. Cheap -- the same profile as
# steer_beta_qwen.sbatch. NOTE the n~58 ceiling: ablation can only act on
# baseline caves, and the factual test split (fixed 95/5) yields ~58 of them. This
# is the necessity test AT THE RIGHT LAYER, still power-limited by that n.

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: needs the --betas flag on steer.py. Before submitting, on a login node
# in $REPO: `git fetch && git checkout sycophancy_v2 && git pull`. meta.json records
# git_sha.
#
# PROBE FILE: results/probes/probe_L20.npz is written by step 1 and is gitignored --
# it must already be present in $REPO (it is, if step 1 ran here on OSC on the same
# card). If missing, run dump_l20_direction_qwen_v100.sbatch first.

export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1

# confirm the card BEFORE loading 14.6GB of weights -- 7B fp16 OOMs on a 16GB GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# --split test = fresh items the L20 direction was NOT built from (clean necessity
# test). --forced-choice matches the DID_SYSTEM regime the direction was built in.
# --betas dense near 1 (full removal, where a real necessity signal shows) plus an
# overcorrection tail. For the full causal panel add --mode both (adds the powered
# ~761-item add pass); that roughly doubles cost -- bump walltime to ~03:30:00.
python -m scripts.steer --probe results/probes/probe_L20.npz --model Qwen/Qwen2.5-7B-Instruct --source factual --split test --mode ablate --control random --forced-choice --max-new-tokens 16 --betas 0,0.5,1,1.5,2,3,4,6,8
