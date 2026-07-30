#!/bin/bash
# STEP 1 of 2 -- dump the L20 (and L19) DiD steering direction, Qwen2.5-7B,
# Pitzer V100-32GB (fp16). Pairs with steer_ablate_l20_qwen_v100.sbatch (step 2).
#
# The steer-LAYER sweep (steer_layer_sweep_qwen_v100.sbatch) reports its best
# STEERING layer (L20, ~5 below the L24 detection peak) but never PERSISTS the
# per-layer directions -- they live only in memory during the run. scripts/steer.py
# ablate/add needs a probe.npz on disk. This runs the SAME sweep pipeline with
# --dump-layers 19,20 --dump-only: it labels the train set, extracts the DiD arrows,
# rebuilds the raw diff-in-means direction at L19+L20, writes each as
# probe_L<L>.npz, and EXITS before the (expensive) per-(layer,alpha) causal-eval
# loop. So this is the sweep's front half only -- no add/hold generation.
#
#   sbatch scripts/osc/dump_l20_direction_qwen_v100.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# WHY V100 fp16 (NOT the A100): a direction is model+DTYPE-specific, and STEP 2
# ablates THIS probe -- the two steps MUST share a card/dtype or the ablation acts
# on activations the direction was not built in (the steer.py cross-dtype caveat).
# _pick_dtype gives fp16 on a V100 / bf16 on an A100; L20 was recommended by the
# fp16 V100 run (23-19-47Z, p0331), so fp16 here also reproduces that direction.
# Do NOT run step 1 on an A100 and step 2 on a V100 (or vice versa). meta.json
# records model_dtype -- confirm fp16.
#
# 32GB IS REQUIRED (7B fp16 weights ~14GB + KV cache OOMs a 16GB V100); the
# gpu-exp partition on pitzer schedules the 32GB V100s. If the partition rejects
# the request, discover the right 32GB feature with
# `sinfo -M pitzer -p gpu -N -o "%N %f %G"` and pin it with --constraint.
#SBATCH --job-name=dump-l20-qwen7b-v100
#SBATCH --account=PAS2324
#SBATCH --clusters=pitzer
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime 4h. --dump-only DROPS the sweep's eval loop (12 layers x 19 alphas x
# ~761-item cave generation), so this is much cheaper than the full layer sweep --
# but the DOMINANT cost, labeling the full dual-ordering factual set (~14.8k
# questions) with the sampled gate (gate-n generate calls each), remains (same
# bottleneck as did_qwen.sbatch). To cut it, cap with `--max-train N` (fewer
# labeled questions -> a noisier direction; keep N in the thousands).

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: needs the --dump-layers/--dump-only flags on steer_layer_sweep.
# Before submitting, on a login node in $REPO:
# `git fetch && git checkout sycophancy_v2 && git pull`. meta.json is NOT written
# by a --dump-only run (it exits before the write), so git_sha is not recorded
# here -- confirm the checked-out commit includes --dump-layers before submitting.

export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1   # keeps 14GB of download bars out of the log

# confirm the card BEFORE loading 14.6GB of weights -- 7B fp16 OOMs on a 16GB GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# L*=24 is the DiD detection peak (14-01-53Z run); the sweep band is L*-8..L*+3, so
# L19 and L20 are in band. --gate sampled --gate-n 10 reproduces the labeling that
# recommended L20 (23-19-47Z). did carries DID_SYSTEM by construction, so no
# --forced-choice is needed. No --alphas: --dump-only never reaches the alpha loop.
python -m scripts.steer_layer_sweep --best-layer 24 --direction did --source factual --gate sampled --gate-n 10 --max-new-tokens 16 --model Qwen/Qwen2.5-7B-Instruct --dump-layers 19,20 --dump-only

# STABLE HANDOFF: --dump-layers writes probe_L<L>.npz into the run's timestamped
# dir. Copy the just-written probes to a fixed path so step 2 needs no run-dir name.
# (set -eo pipefail: an empty glob -> ls errors -> the job fails loudly, which is
# the right behaviour if the dump did not produce a probe.)
mkdir -p "$REPO/results/probes"
cp "$(ls -t "$REPO"/results/runs/*/probe_L20.npz | head -1)" "$REPO/results/probes/probe_L20.npz"
cp "$(ls -t "$REPO"/results/runs/*/probe_L19.npz | head -1)" "$REPO/results/probes/probe_L19.npz"
echo "dumped -> $REPO/results/probes/probe_L20.npz (and probe_L19.npz)"
