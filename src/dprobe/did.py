"""Difference-of-differences arrow extraction (ADR 0012, design="did").

`did_filter` (in data/sycophancy.py) assigns each question a caved/held label.
This module turns those labeled questions into **pressure-response arrows** at two
read positions, so the existing layer-sweep + report path can be run on the
arrows directly (arrows ARE the features; `fit_mms` on them is the capitulation
direction -- CONTEXT.md).

For one question we read four activations -- {calm, pressured} x {prompt-final,
answer-token} -- and subtract within each position:

    arrow_promptfinal = act(pressure prompt, no answer)   - act(neutral prompt, no answer)
    arrow_answertoken = act(pressure prompt, committed X) - act(neutral prompt, honest X)

The first subtraction cancels question content; averaging caved arrows minus held
arrows (done later by `fit_mms`) cancels the generic pressure response, leaving
the capitulation direction. See ADR 0012 for why we read BOTH positions (the
answer-token arm carries a letter-identity term the prompt-final arm does not, so
their AUROC gap measures the letter shortcut) and why the answer-token read uses
the bare committed " (X)" string (`commit`-style, ADR 0009 addendum).
"""

from __future__ import annotations

import numpy as np

from .activations import extract, verify_read_positions
from .data.base import Example

# the two read positions a did run produces a probe + report for
POSITIONS = ("promptfinal", "answertoken")


def _variants(ex: Example) -> dict[str, Example]:
    """The four read-variant Examples for one labeled did question.

    Calm side always reads the honest answer (the gate guarantees the model
    defaults to it); pressured side reads the answer whose letter it actually
    emitted -- `matching` if it caved (label 1), `not_matching` if it held.
    """
    matching = ex.meta["matching"]
    not_matching = ex.meta["not_matching"]
    pressured_ans = matching if ex.label == 1 else not_matching
    neutral = ex.meta["neutral_user"]
    # calm side: sycophancy strips the user turn (pressure lives there);
    # sandbagging swaps the SYSTEM prompt instead -- meta['calm_system']
    # carries the control system, and neutral == ex.user (same question text).
    calm_system = ex.meta.get("calm_system", ex.system)
    return {
        "calm_promptfinal": Example(calm_system, neutral, ex.label, ex.deception_type),
        "pressured_promptfinal": Example(ex.system, ex.user, ex.label, ex.deception_type),
        "calm_answertoken": Example(calm_system, neutral, ex.label, ex.deception_type,
                                    meta={"assistant_prefix": not_matching}),
        "pressured_answertoken": Example(ex.system, ex.user, ex.label, ex.deception_type,
                                         meta={"assistant_prefix": pressured_ans}),
    }


def extract_arrows(model, tokenizer, device, examples: list[Example],
                   batch_size=None, acts_dtype=np.float32):
    """Extract pressure-response arrows for both read positions.

    Returns:
        arrows: dict position -> float array [n_questions, n_layers+1, hidden],
                the pressured-minus-calm arrow at that position
        labels: int array [n_questions]  (1 = caved, 0 = held)
        groups: list[str]                the pressure prompt per question (split key)
    """
    if not examples:
        raise ValueError("extract_arrows got no labeled examples")

    keys = ("calm_promptfinal", "pressured_promptfinal",
            "calm_answertoken", "pressured_answertoken")
    variants = [_variants(ex) for ex in examples]

    # guard the answer-token read: assert the committed letter is the last token
    verify_read_positions(tokenizer, [v["pressured_answertoken"] for v in variants])

    # one extraction pass over all 4n variants (best batching); extract preserves
    # input order, so the four aligned blocks slice straight back out
    flat = [v[k] for k in keys for v in variants]
    acts, _ = extract(model, tokenizer, flat, device,
                      batch_size=batch_size, acts_dtype=acts_dtype)
    n = len(examples)
    blocks = {k: acts[i * n:(i + 1) * n] for i, k in enumerate(keys)}

    arrows = {
        "promptfinal": (blocks["pressured_promptfinal"].astype(np.float32)
                        - blocks["calm_promptfinal"].astype(np.float32)),
        "answertoken": (blocks["pressured_answertoken"].astype(np.float32)
                        - blocks["calm_answertoken"].astype(np.float32)),
    }
    labels = np.array([ex.label for ex in examples])
    groups = [ex.user for ex in examples]
    return arrows, labels, groups
