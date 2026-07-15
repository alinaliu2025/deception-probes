#!/bin/bash
# DiD (ADR 0012) version of scripts.train_one for OSC — sandbagging port.
# Submit from the Ascend login node, ladder rungs 29 -> 400 -> 3000:
#   sbatch scripts/osc/train_did.sbatch --pressure incentive --max-examples 29
# Extra flags pass through to train_one. Override model/type per submit:
#   sbatch --export=ALL,DPROBE_TYPE=sycophancy scripts/osc/train_did.sbatch --max-examples 3000
#
# GUARD: refuses to run without --max-examples (a bare --source factual run
# is the full ARC pool, 7k+ questions).

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-did
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

if [[ "$*" != *"--max-examples"* ]]; then
    echo "REFUSING to run: pass --max-examples (ladder: 29 / 400 / 3000)." >&2
    echo "Without it, --source factual loads the full ARC pool (7k+ questions)." >&2
    exit 1
fi

module load miniconda3/24.1.2-py310   # adjust to `module spider miniconda3` output
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DPROBE_TYPE="${DPROBE_TYPE:-sandbagging}"

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — type: $DPROBE_TYPE — design: did"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTE: --model/--type passed explicitly; train_one does NOT read the env vars
# (config default is the 0.5B — the 2026-07-09 silent-default trap)
python -m scripts.train_one --model "$DPROBE_MODEL" \
    --type "$DPROBE_TYPE" --design did --source factual --filter "$@"
