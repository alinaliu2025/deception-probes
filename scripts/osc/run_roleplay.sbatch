#!/bin/bash
# Run the roleplay forced-choice set through Qwen on OSC.
# Wraps src/dprobe/scheming/run_roleplay_through_model.py.
#
#   # smoke test, 20 prompts
#   sbatch scripts/osc/run_roleplay.sbatch --limit 20
#
#   # full 371
#   sbatch scripts/osc/run_roleplay.sbatch
#
#   # reuse an already-built jsonl instead of re-running the parser
#   sbatch scripts/osc/run_roleplay.sbatch --skip-parse
#
# Extra flags pass straight through. Model defaults to the 7b alias; override:
#   sbatch --export=ALL,DPROBE_MODEL=0.5b scripts/osc/run_roleplay.sbatch --limit 20
#
# --full is passed by DEFAULT here, not by accident: without it the parser omits
# honest_label and the script cannot compute a deception rate, so the job would
# burn a GPU allocation to produce letters with nothing to score them against.

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-roleplay
#SBATCH --time=01:00:00
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
export DPROBE_MODEL="${DPROBE_MODEL:-7b}"

cd /fs/ess/PAS2324/dprobe-alina

if [[ ! -f data/roleplaying/dataset.yaml ]]; then
    echo "REFUSING to run: data/roleplaying/dataset.yaml is missing." >&2
    echo "Compute nodes have no internet. On a LOGIN node:" >&2
    echo "  mkdir -p data/roleplaying" >&2
    echo "  curl -L -o data/roleplaying/dataset.yaml \\" >&2
    echo "    https://raw.githubusercontent.com/ApolloResearch/deception-detection/refs/heads/main/data/roleplaying/dataset.yaml" >&2
    exit 1
fi

mkdir -p runs

echo "Job $SLURM_JOB_ID on $(hostname) — model alias: $DPROBE_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --parser-script must be the real path: the script's default is a bare
# filename, which only resolves if you happen to be cd'd into its directory.
python src/dprobe/scheming/run_roleplay_through_model.py \
    --model "$DPROBE_MODEL" \
    --parser-script src/dprobe/scheming/scheming_data_parser.py \
    --full \
    --batch-size 16 \
    "$@"

echo "--- output ---"
ls -la runs/
