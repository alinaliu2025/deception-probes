#!/bin/bash
# Steering causal validation (ADR 0010) on OSC. All flags pass through, e.g.:
#   sbatch scripts/osc/steer.sbatch --probe results/runs/<run>/probe.npz \
#       --control random [--pressure incentive]
# The probe type (sycophancy/sandbagging) is read from the npz itself.

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-steer
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

module load miniconda3/24.1.2-py310
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m scripts.steer --model "$DPROBE_MODEL" --source factual "$@"
