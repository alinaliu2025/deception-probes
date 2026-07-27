#!/bin/bash
# Forced-choice roleplay scheming on OSC: does Qwen2.5-7B pick the deceptive
# completion? All 371 Apollo scenarios.
#
#   # 1. position-bias control FIRST, both orders, 1 greedy sample each
#   sbatch scripts/osc/roleplay_choice.sbatch --both-orders --samples 1
#
#   # 2. only if step 1 shows the rate actually differs by content:
#   sbatch scripts/osc/roleplay_choice.sbatch --both-orders --samples 20
#
# Extra flags pass through. Override the model per submit:
#   sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/roleplay_choice.sbatch --both-orders
#
# WHY --both-orders IS IN EVERY EXAMPLE. With (A) fixed to the honest option,
# "chose (B)" and "chose deceptive" are the same event, so the deception rate
# cannot be separated from the model's positional preference. The 0.5B smoke run
# on 2026-07-27 returned deception_rate 1.0 with picked_A_rate 0.0 -- five out of
# five (B) -- which is exactly what a recency-biased model looks like and exactly
# what real deception looks like. Running both orders costs one extra forward
# pass per scenario and turns that ambiguity into a number: if the deception rate
# is the same in both orders, it is content; if it tracks the letter, it is bias.
#
# Needs data/roleplaying/dataset.yaml. Compute nodes have no internet, so on a
# LOGIN node first:
#   cd /fs/ess/PAS2324/dprobe-alina
#   mkdir -p data/roleplaying
#   curl -L -o data/roleplaying/dataset.yaml \
#     https://raw.githubusercontent.com/ApolloResearch/deception-detection/refs/heads/main/data/roleplaying/dataset.yaml

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-roleplay-choice
#SBATCH --time=01:30:00
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

if [[ ! -f data/roleplaying/dataset.yaml ]]; then
    echo "REFUSING to run: data/roleplaying/dataset.yaml is missing." >&2
    echo "Compute nodes have no internet. Download it on a login node; the" >&2
    echo "curl command is in the header of this file." >&2
    exit 1
fi

if [[ "$*" != *"--both-orders"* && "$*" != *"--swap-order"* ]]; then
    echo "WARNING: neither --both-orders nor --swap-order passed." >&2
    echo "With (A) fixed to honest, the deception rate you get back cannot be" >&2
    echo "distinguished from a positional preference for (B). Proceeding, but" >&2
    echo "do not report the number without the other order to compare against." >&2
fi

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — roleplay forced choice"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --model passed explicitly: the config default is the 0.5B (the 2026-07-09
# silent-default trap), and the script only WARNS about it, it does not refuse
python -m dprobe.scheming.roleplay_choice --model "$DPROBE_MODEL" "$@"
