#!/bin/bash
# DiD cross-type transfer matrix (ADR 0012) for OSC.
# Scores each did probe on every type's did ARROWS at the probe's own layer.
# Pass one --probe per row/col; use the SAME read position for all (default
# prompt-final). The .npz files are gitignored -- scp them back from their
# training runs (or run this on OSC where they already live).
#
#   sbatch scripts/osc/crosstype_did.sbatch \
#       --probe syco=results/runs/<syco-did>/probe_promptfinal.npz \
#       --probe sand-inst=results/runs/<sand-inst-did>/probe_promptfinal.npz
#
# Add a sandbagging arm by @pressure, e.g. sand-ince=<run>/probe_promptfinal.npz@incentive

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-xtype-did
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

if [[ "$*" != *"--probe"* ]]; then
    echo "REFUSING to run: pass at least one --probe label=path/to/probe_promptfinal.npz" >&2
    exit 1
fi

module load miniconda3/24.1.2-py310   # adjust to `module spider miniconda3` output
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cd /fs/ess/PAS2324/dprobe-alina

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — DiD cross-type transfer"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# each distinct (type, pressure) pays one belief-gate + arrow-extraction pass;
# probes of the same arm share it. Default position is prompt-final (clean arm).
python -m scripts.crosstype_did --model "$DPROBE_MODEL" --source factual "$@"
