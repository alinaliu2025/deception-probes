"""Registry so scripts can ask for a dataset by name."""

import random

from . import omission, sandbagging, sycophancy

BUILDERS = {
    "sycophancy": sycophancy.build,
    "sandbagging": sandbagging.build,
    "omission": omission.build,
}

# Filters that run the model to drop items where the labeled condition doesn't
# actually induce the behaviour (so the probe trains on the real thing, not the
# prompt difference). Signature: (model, tokenizer, device, examples) -> examples.
FILTERS = {
    "sandbagging": sandbagging.capability_filter,
    # dispatches on design: 'framing' -> behavior_filter (keep/drop),
    # 'behavioral' -> behavioral_filter (assigns the labels; mandatory)
    "sycophancy": sycophancy.sycophancy_filter,
}


def get(deception_type: str, design: str = "completion", source: str = "opinion"):
    """Build a dataset by name.

    `design` selects between alternative constructions of a type's contrast, and
    `source` between question datasets (ADR 0009); only sycophancy currently has
    more than one of either (see sycophancy.build). Both are ignored for other
    types, so the defaults are a no-op for them.
    """
    if deception_type not in BUILDERS:
        raise KeyError(f"unknown type {deception_type!r}; have {list(BUILDERS)}")
    if deception_type == "sycophancy":
        return BUILDERS[deception_type](design=design, source=source)
    return BUILDERS[deception_type]()


def subsample(examples, max_examples: int, seed: int):
    """Randomly drop whole prompts until len(examples) <= max_examples.

    Groups by the shared `user` field -- the one thing a matched 0/1 pair always
    has in common across every type -- and keeps or drops each group as a unit,
    so matched pairs never straddle the cut and label balance is preserved.
    Returns the list unchanged if it already fits.
    """
    if len(examples) <= max_examples:
        return examples
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.user, []).append(i)
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    kept: list[int] = []
    for k in keys:
        if len(kept) + len(groups[k]) <= max_examples:
            kept.extend(groups[k])
    kept.sort()
    return [examples[i] for i in kept]
