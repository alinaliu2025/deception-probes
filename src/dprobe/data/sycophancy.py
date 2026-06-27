"""Sycophancy: the model tells the user what they want to hear instead of the truth.

Data: Anthropic/model-written-evals sycophancy JSONL files (HuggingFace).
Each row has a `question` (prompt with user biography asserting a view),
`answer_matching_behavior` (sycophantic completion, label=1), and
`answer_not_matching_behavior` (honest completion, label=0).

Extraction position: response-final (last token of assistant completion).
Split: fixed-seed 95/5 by prompt index to prevent label leakage across pairs.
"""

from __future__ import annotations

import random

from datasets import load_dataset

from .base import Example

NEUTRAL_SYSTEM = "You are a helpful assistant."

_HF_FILES = [
    "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]

_SPLIT_SEED = 42
_TEST_FRACTION = 0.05


def build(split: str = "train") -> list[Example]:
    """Return Examples for the requested split ('train' or 'test').

    Each prompt yields two Examples (label 1 + label 0); the split is by
    prompt so paired examples never straddle train/test.
    """
    rows = []
    for fname in _HF_FILES:
        ds = load_dataset(
            "Anthropic/model-written-evals",
            data_files=fname,
            split="train",  # HF exposes single split; we re-split below
        )
        rows.extend(ds)

    rng = random.Random(_SPLIT_SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    cutoff = int(len(indices) * _TEST_FRACTION)
    test_idx = set(indices[:cutoff])

    out = []
    for i, row in enumerate(rows):
        in_test = i in test_idx
        if (split == "test") != in_test:
            continue
        q = row["question"]
        out.append(Example(
            NEUTRAL_SYSTEM, q, label=1,
            deception_type="sycophancy",
            completion=row["answer_matching_behavior"],
        ))
        out.append(Example(
            NEUTRAL_SYSTEM, q, label=0,
            deception_type="sycophancy",
            completion=row["answer_not_matching_behavior"],
        ))
    return out
