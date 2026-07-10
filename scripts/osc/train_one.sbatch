#!/bin/bash
# Batch version of scripts.train_one for OSC (ADR 0002).
# Submit from the Ascend login node (A100 = formal tier):
#   sbatch scripts/osc/train_one.sbatch
# Extra flags pass through to train_one, e.g.:
#   sbatch scripts/osc/train_one.sbatch --rollouts 16 --gate sampled
# Override the model per submit:
#   sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/train_one.sbatch

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

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m scripts.train_one --type sycophancy --design rollout --source factual --filter "$@"
