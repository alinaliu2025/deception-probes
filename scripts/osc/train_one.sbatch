#!/bin/bash
# Batch version of scripts.train_one for OSC (ADR 0002).
# Submit from the Ascend login node (A100 = formal tier):
#   sbatch scripts/osc/train_one.sbatch
# Extra flags pass through to train_one, e.g.:
#   sbatch scripts/osc/train_one.sbatch --rollouts 16 --gate sampled
# Override the model per submit:
#   sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/train_one.sbatch
# Override the deception type per submit (ADR 0011); pass type-specific flags after:
#   sbatch --export=ALL,DPROBE_TYPE=sandbagging scripts/osc/train_one.sbatch \
#       --pressure incentive --max-new-tokens 48

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

module load miniconda3/24.1.2-py310   # adjust to `module spider miniconda3` output
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DPROBE_TYPE="${DPROBE_TYPE:-sycophancy}"

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — type: $DPROBE_TYPE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: --model/--type must be passed explicitly; train_one does NOT read the
# env vars ($DPROBE_MODEL: the 2026-07-09 p0256 run silently used the 0.5B
# default because of this)
python -m scripts.train_one --model "$DPROBE_MODEL" \
    --type "$DPROBE_TYPE" --design rollout --source factual --filter "$@"
