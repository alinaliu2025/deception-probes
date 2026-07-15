#!/bin/bash
# Steering / causal validation (ADR 0010) for OSC — DiD probes.
# The probe's TYPE and ARM are read from the .npz; steer under the SAME regime it
# was trained in (forced-choice for did), so this always passes --forced-choice.
#
# Sandbagging (instructed arm), direction from a scp'd-back run (type is read
# from the .npz; --pressure must match the arm the probe was trained on):
#   sbatch scripts/osc/steer.sbatch \
#       --probe results/runs/<run>/probe_promptfinal.npz --pressure instructed
#
# Sycophancy:
#   sbatch scripts/osc/steer.sbatch --probe results/runs/<run>/probe_promptfinal.npz
#
# --probe is REQUIRED (guarded below). Steer the prompt-final probe, not
# answer-token (ADR 0012: answer-token carries the letter-identity term).

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-steer
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

if [[ "$*" != *"--probe"* ]]; then
    echo "REFUSING to run: pass --probe results/runs/<run>/probe_promptfinal.npz" >&2
    echo "The .npz is gitignored -- scp it back from the training run first." >&2
    exit 1
fi

module load miniconda3/24.1.2-py310   # adjust to `module spider miniconda3` output
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — steering (forced-choice)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# forced-choice: measure caving in the same bare "(A)/(B)" regime the did probe
# was trained in (small token cap so generation reaches the commit). Type + arm
# come from the probe / passthrough flags. --control random is the baseline that
# separates DIRECTION from generic perturbation -- keep it for any causal claim.
python -m scripts.steer --model "$DPROBE_MODEL" --source factual \
    --forced-choice --max-new-tokens 16 --control random "$@"
