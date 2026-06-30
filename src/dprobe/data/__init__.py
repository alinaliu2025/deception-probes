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
    "sycophancy": sycophancy.behavior_filter,  # design="framing" only
}


def get(deception_type: str, design: str = "completion"):
    """Build a dataset by name.

    `design` selects between alternative constructions of a type's contrast; only
    sycophancy currently has more than one (see sycophancy.build). It is ignored
    for types with a single design, so the default is a no-op for them.
    """
    if deception_type not in BUILDERS:
        raise KeyError(f"unknown type {deception_type!r}; have {list(BUILDERS)}")
    if deception_type == "sycophancy":
        return BUILDERS[deception_type](design=design)
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
