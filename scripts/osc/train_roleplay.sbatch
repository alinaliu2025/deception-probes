#!/bin/bash
# Apollo roleplay scheming probe on OSC (Qwen2.5-7B-Instruct).
#
#   sbatch scripts/osc/train_roleplay.sbatch
#   sbatch scripts/osc/train_roleplay.sbatch --method mms
#   sbatch scripts/osc/train_roleplay.sbatch --permute     # leak control
#
# Extra flags pass through to train_one.
#
# NO GENERATION HAPPENS HERE. The roleplay design is teacher-forced: Apollo
# supplies both the honest and the deceptive completion, each is appended as if
# the model had said it, and the hidden state is read at the response-final
# token. 742 forward passes on 371 pairs. Minutes, not hours.
#
# GUARD: DPROBE_STRICT_ROLEPLAY=1 makes a missing yaml a hard failure. Without
# it, `build_roleplay` silently falls back to the 10 items inlined in
# scheming.py and you get a probe trained on a smoke-test set with no error.
# Download the yaml on a LOGIN node first, compute nodes have no internet:
#
#   cd /fs/ess/PAS2324/dprobe-alina
#   mkdir -p data/roleplaying
#   curl -L -o data/roleplaying/dataset.yaml \
#     https://raw.githubusercontent.com/ApolloResearch/deception-detection/main/data/roleplaying/dataset.yaml
#   python -m scripts.bow_roleplay --yaml data/roleplaying/dataset.yaml

#SBATCH --account=PAS2324
#SBATCH --job-name=dprobe-roleplay
#SBATCH --time=00:40:00
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
export DPROBE_STRICT_ROLEPLAY=1

cd /fs/ess/PAS2324/dprobe-alina

if [[ ! -f data/roleplaying/dataset.yaml ]]; then
    echo "REFUSING to run: data/roleplaying/dataset.yaml is missing." >&2
    echo "Compute nodes have no internet. Download it on a login node first," >&2
    echo "see the header of this file for the curl command." >&2
    exit 1
fi

echo "Job $SLURM_JOB_ID on $(hostname) — model: $DPROBE_MODEL — type: scheming — design: roleplay"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import yaml,pathlib;d=yaml.safe_load(pathlib.Path('data/roleplaying/dataset.yaml').read_text());print('yaml items:',len(d) if isinstance(d,list) else 'WRAPPED, keys='+str(sorted(d)))"

# --model passed explicitly: train_one does NOT read the env var, and the config
# default is the 0.5B (the 2026-07-09 silent-default trap)
python -m scripts.train_one --model "$DPROBE_MODEL" \
    --type scheming --design roleplay "$@"
