#!/bin/bash
# DiD sycophancy run, OLMo-2 7B, Ascend A100 (ADR 0002 formal path).
#
#   sbatch scripts/osc/did.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=ascend <jobid>
#
# Batch, not interactive: an salloc/sinteractive shell dies with the SSH
# connection and takes the run with it. This survives a disconnect.
#
# Pitzer V100-32GB instead of Ascend: drop the --cluster/--partition lines and
# use `#SBATCH --constraint=48core`. Note that changes the dtype -- _pick_dtype
# gives fp16 on V100 (compute 7.0) and bf16 on A100 (8.0); meta.json records
# which one ran as `model_dtype`.
#SBATCH --job-name=did-olmo7b
#SBATCH --account=PAS2324
#SBATCH --cluster=ascend
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# per-user cache: a shared HF_HOME fails with PermissionError the first time
# someone pulls a model its owner didn't
export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1   # keeps 14GB of download bars out of the log

# confirm the card BEFORE loading 14.6GB of weights -- 7B fp16 OOMs on a 16GB GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m scripts.train_one --type sycophancy --design did --source factual --filter --method mms --gate-n 10 --max-new-tokens 16 --model allenai/OLMo-2-1124-7B-Instruct
