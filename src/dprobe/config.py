"""Central knobs. Change the model here once and everything downstream follows."""

import os
from pathlib import Path

# Small, ungated, CPU-friendly default for local smoke runs; OSC overrides it.
# Precedence: --model flag > $DPROBE_MODEL > this default (ADR 0002).
# Qwen is the project default (Phase 1 lock; the OSC/DiD runs are all Qwen-7B).
LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OSC_MODEL = "Qwen/Qwen2.5-7B-Instruct"

MODEL_NAME = os.environ.get("DPROBE_MODEL", LOCAL_MODEL)

TEST_FRAC = 0.3
SEED = 0

# the three behaviours we study this week, all single turn
DECEPTION_TYPES = ["sycophancy", "sandbagging", "omission"]

# repo-root/results, created if missing
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
