"""Central knobs. Change the model here once and everything downstream follows."""

from pathlib import Path

# ~1GB, not gated, runs on a laptop CPU/MPS. Bump to Qwen2.5-1.5B-Instruct once
# the pipeline works end to end and you have OSC GPU time.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEST_FRAC = 0.3
SEED = 0

# the three behaviours we study this week, all single turn
DECEPTION_TYPES = ["sycophancy", "sandbagging", "omission"]

# repo-root/results, created if missing
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
