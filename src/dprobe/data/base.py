"""The contract every dataset module follows.

An Example is ONE prompt with a label. A dataset is a list of Examples, balanced
between label 0 (control / honest) and label 1 (the deceptive behaviour). Each
deception type lives in its own module and exposes build() -> list[Example].

Design rule: the ONLY thing that should differ between a matched control and
deceptive Example is whatever induces the behaviour (the system prompt, or the
user's framing). Keep the underlying task identical so the probe learns the
behaviour, not the topic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Example:
    system: str
    user: str
    label: int               # 1 = deceptive condition, 0 = control
    deception_type: str
    meta: dict = field(default_factory=dict)  # e.g. correct answer, for capability filtering


def paired(deception_type: str, items, control_system: str, deceptive_system: str):
    """Convenience: turn a list of user-prompt strings into matched 0/1 Examples
    that differ only in the system prompt. Good enough for a first pass; for
    stronger designs vary the user framing instead (see sycophancy)."""
    out = []
    for user in items:
        out.append(Example(control_system, user, 0, deception_type))
        out.append(Example(deceptive_system, user, 1, deception_type))
    return out
