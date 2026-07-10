"""Make the repo root importable so tests can import from scripts/.

`dprobe` is pip-installed (-e), but `scripts/` is deliberately not a package --
train_one etc. run via `python -m scripts.<name>` from the repo root. Tests that
import script internals (tests/test_crossread.py) need the root on sys.path,
which pytest does not add when tests/ has no __init__.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
