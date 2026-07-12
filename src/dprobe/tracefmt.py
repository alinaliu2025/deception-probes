"""Shared formatting vocabulary for filter traces (``run_dir/run_log.txt``).

A *filter trace* is the structured per-question record of every keep / drop /
relabel decision a behaviour filter or belief gate made during a run (see
CONTEXT.md). Each filter renders its own trace next to its logic; this module
only holds the tiny helpers they share so every trace reads the same way.
"""

from __future__ import annotations

import json

WIDTH = 78


def short(q: str, n: int = 90) -> str:
    """Whitespace-collapse and truncate a question for a one-line label."""
    q = " ".join(q.split())
    return q if len(q) <= n else q[: n - 1] + "…"


def qtext(t: str) -> str:
    """A generated text as ONE unambiguous log line (quoted, escapes newlines)."""
    return json.dumps(t, ensure_ascii=False)


def render(title: str, stats: dict | None,
           sections: list[tuple[str, list[str]]]) -> str:
    """Assemble a trace: banner title, SUMMARY block, then titled sections.

    Section body lines are indented two spaces; pass them unindented.
    """
    bar = "=" * WIDTH
    lines = [bar, title, bar, ""]
    if stats:
        lines.append("SUMMARY")
        lines += [f"  {k}: {v}" for k, v in stats.items()]
        lines.append("")
    for heading, body in sections:
        lines.append(heading)
        lines += [f"  {ln}" for ln in body]
        lines.append("")
    lines.append(bar)
    return "\n".join(lines)
