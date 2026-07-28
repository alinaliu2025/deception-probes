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
# Extra flags pass straight through. Model defaults to the FULL 7B id; override:
#   sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-0.5B-Instruct scripts/osc/run_roleplay.sbatch --limit 20
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
# 48-core Pitzer nodes carry 2x V100 32GB; the 40-core ones carry 16GB cards.
# Qwen2.5-7B in 16-bit is ~15.2GB of weights, so on a 16GB card there is roughly
# 200MB left and attention OOMs immediately (job 50773891). docs/HPC.md line 29.
#SBATCH --constraint=48core
#SBATCH --output=results/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alinaliu2025@gmail.com

set -euo pipefail

module load miniconda3/24.1.2-py310   # adjust to `module spider miniconda3` output
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina

export HF_HOME=/fs/scratch/PAS2324/hf

# Full HF id, not the "7b" alias. The runner's own DEFAULT_MODEL is "0.5b", so
# every layer of indirection between here and AutoModelForCausalLM is a chance
# to silently burn a GPU allocation on the wrong model (2026-07-09). Spell it out.
export DPROBE_MODEL="${DPROBE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cd /fs/ess/PAS2324/dprobe-alina

if [[ ! -f data/roleplaying/dataset.yaml ]]; then
    echo "REFUSING to run: data/roleplaying/dataset.yaml is missing." >&2
    echo "Compute nodes have no internet. On a LOGIN node:" >&2
    echo "  mkdir -p data/roleplaying" >&2
    echo "  curl -L -o data/roleplaying/dataset.yaml \\" >&2
    echo "    https://raw.githubusercontent.com/ApolloResearch/deception-detection/refs/heads/main/data/roleplaying/dataset.yaml" >&2
    exit 1
fi

if [[ "$DPROBE_MODEL" == *"0.5B"* || "$DPROBE_MODEL" == "0.5b" ]]; then
    if [[ "$*" != *"--limit"* ]]; then
        echo "REFUSING to run: the 0.5B on the FULL set." >&2
        echo "It picked (B) 5/5 in both letter orders on 2026-07-27, i.e. pure" >&2
        echo "position bias with no signal. Add --limit for a plumbing check, or" >&2
        echo "override DPROBE_MODEL if you really mean it." >&2
        exit 1
    fi
    echo "NOTE: 0.5B with --limit. Plumbing check only, not a deception rate."
fi

mkdir -p runs

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail fast and legibly instead of 90 seconds into generation with a traceback.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ "$DPROBE_MODEL" == *"7B"* && "$VRAM_MIB" -lt 20000 ]]; then
    echo "REFUSING to run: ${VRAM_MIB} MiB of VRAM, need >20000 for a 7B." >&2
    echo "7B in 16-bit is ~15.2GB of weights; a 16GB card leaves nothing for" >&2
    echo "attention. Submit with --constraint=48core (2x V100 32GB) or move to" >&2
    echo "Ascend for an A100. See docs/HPC.md." >&2
    exit 1
fi

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
