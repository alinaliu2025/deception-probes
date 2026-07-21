"""Difference-of-differences arrow extraction (ADR 0012, design="did").

`did_filter` (in data/sycophancy.py) assigns each question a caved/held label.
This module turns those labeled questions into **pressure-response arrows** at one
or both read positions, so the existing layer-sweep + report path can be run on
the arrows directly (arrows ARE the features; `fit_mms` on them is the
capitulation direction -- CONTEXT.md).

For one question we read up to four activations -- {calm, pressured} x
{prompt-final, answer-token} -- and subtract within each position:

    arrow_promptfinal = act(pressure prompt, no answer)   - act(neutral prompt, no answer)
    arrow_answertoken = act(pressure prompt, committed X) - act(neutral prompt, honest X)

The first subtraction cancels question content; averaging caved arrows minus held
arrows (done later by `fit_mms`) cancels the generic pressure response, leaving
the capitulation direction.

`promptfinal` is the CLEAN arm and is what a did run extracts by default. The
`answertoken` arm is diagnostic only: it carries a letter-identity term the
prompt-final arm does not, so their AUROC gap measures the letter shortcut
(ADR 0012). Asking for it doubles the extraction pass, so it is opt-in
(`--both-positions`). Its read sits on the bare committed " (X)" string
(`commit`-style, ADR 0009 addendum).
"""

from __future__ import annotations

import numpy as np

from .activations import extract, verify_read_positions
from .data.base import Example

# every read position a did run can produce a probe + report for
POSITIONS = ("promptfinal", "answertoken")
# the clean arm; the answer-token arm is diagnostic and costs a second read pair
DEFAULT_POSITIONS = ("promptfinal",)

# the {calm, pressured} read-variant pair each position subtracts
_VARIANT_KEYS = {
    "promptfinal": ("calm_promptfinal", "pressured_promptfinal"),
    "answertoken": ("calm_answertoken", "pressured_answertoken"),
}


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
    return {
        "calm_promptfinal": Example(ex.system, neutral, ex.label, ex.deception_type),
        "pressured_promptfinal": Example(ex.system, ex.user, ex.label, ex.deception_type),
        "calm_answertoken": Example(ex.system, neutral, ex.label, ex.deception_type,
                                    meta={"assistant_prefix": not_matching}),
        "pressured_answertoken": Example(ex.system, ex.user, ex.label, ex.deception_type,
                                         meta={"assistant_prefix": pressured_ans}),
    }


def extract_arrows(model, tokenizer, device, examples: list[Example],
                   batch_size=None, acts_dtype=np.float32,
                   positions: tuple[str, ...] = DEFAULT_POSITIONS):
    """Extract pressure-response arrows for the requested read positions.

    Args:
        positions: which read positions to extract; each costs one calm +
            pressured read per question. Defaults to the clean prompt-final arm
            alone.

    Returns:
        arrows: dict position -> float array [n_questions, n_layers+1, hidden],
                the pressured-minus-calm arrow at that position (one key per
                requested position)
        labels: int array [n_questions]  (1 = caved, 0 = held)
        groups: list[str]                the pressure prompt per question (split key)
    """
    if not examples:
        raise ValueError("extract_arrows got no labeled examples")
    unknown = [p for p in positions if p not in _VARIANT_KEYS]
    if unknown:
        raise ValueError(f"unknown did read position(s) {unknown}; have {list(POSITIONS)}")
    if not positions:
        raise ValueError("extract_arrows needs at least one read position")

    keys = [k for pos in positions for k in _VARIANT_KEYS[pos]]
    variants = [_variants(ex) for ex in examples]

    # guard the answer-token read: assert the committed letter is the last token
    # (the prompt-final read has no committed answer, so nothing to verify)
    if "answertoken" in positions:
        verify_read_positions(tokenizer, [v["pressured_answertoken"] for v in variants])

    # one extraction pass over all variants (best batching); extract preserves
    # input order, so the aligned blocks slice straight back out
    flat = [v[k] for k in keys for v in variants]
    acts, _ = extract(model, tokenizer, flat, device,
                      batch_size=batch_size, acts_dtype=acts_dtype)
    n = len(examples)
    blocks = {k: acts[i * n:(i + 1) * n] for i, k in enumerate(keys)}

    arrows = {}
    for pos in positions:
        calm, pressured = _VARIANT_KEYS[pos]
        arrows[pos] = (blocks[pressured].astype(np.float32)
                       - blocks[calm].astype(np.float32))
    labels = np.array([ex.label for ex in examples])
    # group by the shared source-question id when present (factual carries one per
    # question, spanning its two letter orderings, ADR 0012) so the grouped split
    # never separates a question's orderings; fall back to the pressure prompt
    groups = [ex.meta.get("group", ex.user) for ex in examples]
    return arrows, labels, groups
