"""Central knobs. Change the model here once and everything downstream follows."""

import os
from pathlib import Path

# Small, ungated, CPU-friendly default for local smoke runs; OSC overrides it.
# Precedence: --model flag > $DPROBE_MODEL > this default (ADR 0002).
LOCAL_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
OSC_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

MODEL_NAME = os.environ.get("DPROBE_MODEL", LOCAL_MODEL)

TEST_FRAC = 0.3
SEED = 0

# the three behaviours we study this week, all single turn
DECEPTION_TYPES = ["sycophancy", "sandbagging", "omission"]

# repo-root/results, created if missing
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
