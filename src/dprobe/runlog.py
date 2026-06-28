"""Immutable, self-describing run directories.

Every run gets its own folder under ``results/runs/``, named with timestamp +
git SHA + host so two machines never clobber each other. The tiny ``meta.json``
holds all config + metrics in machine-readable form -- that is the file you
commit to git and compare across machines. The heavy ``.npy`` / ``.png``
artifacts beside it stay gitignored and are regenerable from the metadata.

    run_dir = new_run_dir("sycophancy", "lr")
    # ... save figures / arrays into run_dir ...
    write_meta(run_dir, {"type": "sycophancy", "auroc": 0.74, ...})
"""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .config import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "runs"

# versions worth pinning to a number for reproducibility across machines
_PKGS = ("torch", "transformers", "scikit-learn", "numpy")

_PKG_ROOT = Path(__file__).resolve().parent


def _safe(s: str) -> str:
    """Make a string safe to drop into a directory name."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", s)


def _git_sha() -> tuple[str, bool]:
    """(short SHA, dirty?) for the working tree, or ("nogit", False) if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PKG_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_PKG_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip())
        return sha or "nogit", dirty
    except Exception:
        return "nogit", False


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in _PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "n/a"
    return out


def _host() -> str:
    return socket.gethostname().split(".")[0]


def new_run_dir(kind: str, method: str) -> Path:
    """Create and return ``results/runs/<utc>_<kind>_<method>_<sha>_<host>/``.

    The timestamp+sha+host in the name guarantee two runs -- on the same or
    different machines -- never write to the same path.
    """
    sha, _ = _git_sha()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    name = f"{stamp}_{_safe(kind)}_{_safe(method)}_{_safe(sha)}_{_safe(_host())}"
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_meta(run_dir: Path, meta: dict) -> Path:
    """Write ``run_dir/meta.json``, auto-filling environment/provenance fields.

    Caller-supplied keys win over the auto-filled ones, so you can override e.g.
    ``host`` for a recorded run. This is the one file meant to be tracked in git.
    """
    sha, dirty = _git_sha()
    full = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": _host(),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "versions": _versions(),
        **meta,
    }
    out = run_dir / "meta.json"
    out.write_text(json.dumps(full, indent=2, default=_jsonable))
    return out
