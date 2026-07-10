#!/bin/bash
# Batch version of scripts.crossread for OSC: the commit-vs-text 2x2.
# Submit:
#   sbatch scripts/osc/crossread.sbatch
# Extra flags pass through, e.g.:
#   sbatch scripts/osc/crossread.sbatch --type sandbagging --rollouts 16
# Override the model per submit:
#   sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/crossread.sbatch

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-crossread
#SBATCH --time=08:00:00
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

python -m scripts.crossread --model "$DPROBE_MODEL" \
    --type sycophancy --source factual --method mms --max-examples 1500 "$@"
