#!/bin/bash
# Concept-answer DiD sycophancy run (ADR 0013), Qwen2.5-7B, Pitzer V100-32GB (fp16).
#
# The FALLBACK design for the B-pusher shortcut: the pressure asserts a wrong
# CONCEPT (not an (A)/(B) letter) and the model answers one word, so there is no
# letter to shortcut on. Labeled by normalized logprob (decider) + a greedy
# one-word generation cross-check; run_log.txt marks AGREE/DIVERGE per item and
# meta.json carries the divergence rate + cave rate. Compare its promptfinal AUROC
# and steering against the letter run (did_qwen.sbatch).
#
#   sbatch scripts/osc/did_concept_qwen.sbatch
#   squeue -u $USER --clusters=all
#   scancel --clusters=pitzer <jobid>
#
# CAVEAT: the curated concept set is small (~60 questions, no letter orderings) --
# a PLUMBING/first-signal run, NOT a trustworthy AUROC (CLAUDE.md tiny-seed rule).
# Scale src/dprobe/data/fixtures/concept.jsonl before drawing conclusions.
#
# V100 (compute 7.0) matches the letter run's card so model_dtype (fp16) matches
# and the comparison is apples-to-apples. 48core = the 2x V100-32GB node.
#SBATCH --job-name=did-concept-qwen7b
#SBATCH --account=PAS2324
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48core
#SBATCH --time=02:00:00
#SBATCH --output=/fs/scratch/PAS2324/deception-probes/results/slurm-%j.out

# Walltime is 2h, not 8h: concept is ONE ordering of ~60 questions (vs factual's
# ~14.8k after dual ordering), so the sampled gate -- the bottleneck -- is tiny.
# Model download/load dominates; trim once a real run reports its wall time.

set -eo pipefail

REPO=/fs/scratch/PAS2324/deception-probes
cd "$REPO"
source "$REPO/.venv/bin/activate"

# CODE SYNC: this run needs the ADR 0013 concept commit. Before submitting, on a
# login node in $REPO: `git fetch && git checkout sycophancy_v2 && git pull` (or
# the branch/sha you pushed). meta.json records git_sha -- confirm it matches.

export HF_HOME=/fs/scratch/PAS2324/$USER/hf
export PYTHONNOUSERSITE=True
export HF_HUB_DISABLE_PROGRESS_BARS=1

nvidia-smi --query-gpu=name,memory.total --format=csv

python -m scripts.train_one --type sycophancy --design did --source concept --filter --method mms --gate-n 10 --max-new-tokens 16 --model Qwen/Qwen2.5-7B-Instruct
