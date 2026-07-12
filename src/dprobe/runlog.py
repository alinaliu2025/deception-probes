"""Immutable, self-describing run directories.

Every run gets its own folder under ``results/runs/``, named with timestamp +
git SHA + host so two machines never clobber each other. The tiny ``meta.json``
holds all config + metrics in machine-readable form -- that is the file you
commit to git and compare across machines. The heavy ``.npy`` / ``.png``
artifacts beside it stay gitignored and are regenerable from the metadata.

The run dir is created at the START of a run and ``meta.json`` is written last,
so a dir without ``meta.json`` is a crashed/aborted run -- its ``console.log``
is the record of how far it got (ADR 0011).

    run_dir = new_run_dir("sycophancy", "lr")
    with capture_console(run_dir):
        # ... whole run: everything printed lands in console.log too ...
        write_meta(run_dir, {"type": "sycophancy", "auroc": 0.74, ...})
"""

from __future__ import annotations

import io
import json
import platform
import re
import socket
import subprocess
import sys
import traceback
from contextlib import contextmanager
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


class _ConsoleSink:
    """Line assembler for the console tee.

    Timestamps each completed line and collapses ``\\r`` in-place progress
    updates (``print(..., end="\\r")`` counters) to their FINAL state, so the
    log shows "  1200/1200" once instead of thousands of intermediate frames.
    Shared by the stdout and stderr tees so their lines interleave in wall-clock
    order, same as on the terminal.
    """

    _SEP = re.compile(r"\r\n|\n|\r")

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""
        self._progress: str | None = None  # last \r-overwritten frame

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    def raw(self, line: str) -> None:
        """Write a line verbatim (no timestamp) -- for the header/footer."""
        self._fh.write(line + "\n")
        self._fh.flush()

    def feed(self, s: str) -> None:
        self._buf += s
        while True:
            m = self._SEP.search(self._buf)
            if m is None:
                return
            line, sep = self._buf[:m.start()], m.group()
            self._buf = self._buf[m.end():]
            if sep == "\r":
                if line:
                    self._progress = line  # keep only the newest frame
            else:
                self._emit(line)

    def _emit(self, line: str) -> None:
        if self._progress is not None:
            # a progress run just ended: log its final state exactly once. A
            # bare newline after the counter is only its terminator, not a line.
            self._fh.write(f"{self._ts()} {self._progress}\n")
            self._progress = None
            if not line:
                return
        self._fh.write(f"{self._ts()} {line}\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        """Flush any pending progress frame / partial line (crash-safety)."""
        if self._progress is not None:
            self._fh.write(f"{self._ts()} {self._progress}\n")
            self._progress = None
        if self._buf:
            self._fh.write(f"{self._ts()} {self._buf}\n")
            self._buf = ""
        self._fh.flush()


class _Tee(io.TextIOBase):
    """Write-through wrapper: the real stream sees everything unchanged, the
    sink gets a copy. isatty/encoding delegate so progress bars and colour
    detection behave exactly as without the tee."""

    def __init__(self, stream, sink: _ConsoleSink):
        self._stream = stream
        self._sink = sink

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        self._sink.feed(s)
        return len(s) if n is None else n

    def flush(self) -> None:
        self._stream.flush()
        self._sink.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


@contextmanager
def capture_console(run_dir: Path):
    """Tee stdout+stderr into ``run_dir/console.log`` for the duration.

    The terminal/SLURM stream is unchanged; the file gets a UTC-timestamped
    copy of every line (so stage durations are reconstructable), with ``\\r``
    progress counters collapsed to their final state. If the body raises, the
    traceback is written to the log before propagating -- crashed runs are the
    ones most worth recording.
    """
    path = run_dir / "console.log"
    fh = open(path, "a", encoding="utf-8", errors="replace", newline="\n")
    sink = _ConsoleSink(fh)
    start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sink.raw(f"# console log | started {start} | argv: {' '.join(sys.argv)}")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, sink), _Tee(old_err, sink)
    try:
        yield path
    except BaseException:
        sink.feed(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        sink.close()
        end = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sink.raw(f"# console log | ended {end}")
        fh.close()


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
