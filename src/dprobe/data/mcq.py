"""Shared two-choice MCQ source: ARC questions + the offline smoke fixture.

Extracted so sandbagging (and any future type) can build on the same question
pool as sycophancy's `factual` source (ADR 0009) WITHOUT importing sycophancy's
prompt construction. `sycophancy.py` is deliberately left untouched; its
`_factual_rows` duplicates this loading for now and the two are kept
seed-identical so they can be merged later (see ADR 0011).

Invariants shared with sycophancy's factual source, all seeded on
(_SPLIT_SEED, idx) so every consumer agrees:

- upstream HF splits are concatenated and re-split with the SAME fixed-seed
  95/5 index pattern -- a question is in `test` for one deception type iff it
  is in `test` for every other, which is what the cross-type transfer matrix
  needs (no train/test contamination across types);
- the correct/wrong letter side is the SAME per-index coin flip, so the body
  built here is byte-identical to sycophancy's `neutral_user` for that row;
- the distractor is the same seeded-random wrong choice.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# MUST match sycophancy._SPLIT_SEED / _TEST_FRACTION (verified by test)
_SPLIT_SEED = 42
_TEST_FRACTION = 0.05

_ARC_CONFIGS = ("ARC-Easy", "ARC-Challenge")


def two_choice_body(question: str, correct: str, wrong: str, idx: int):
    """Format one two-choice MCQ body, deterministically in `idx`.

    Returns (body, correct_tag, wrong_tag) where the tags are the " (X)" answer
    strings. The letter side comes from the same per-index coin flip as
    sycophancy._make_factual_row, so for a given source row this body equals
    that row's `neutral_user` byte-for-byte (cross-type invariant, tested).
    """
    rng = random.Random(f"{_SPLIT_SEED}:{idx}")
    if rng.random() < 0.5:
        a, b = correct, wrong
        correct_letter, wrong_letter = "A", "B"
    else:
        a, b = wrong, correct
        correct_letter, wrong_letter = "B", "A"
    body = f"{question}\n (A) {a}\n (B) {b}\n\nAnswer:"
    return body, f" ({correct_letter})", f" ({wrong_letter})"


def _test_indices(n: int) -> set[int]:
    """The fixed-seed 5% test index set over n rows (same pattern everywhere)."""
    rng = random.Random(_SPLIT_SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    cutoff = int(len(indices) * _TEST_FRACTION)
    return set(indices[:cutoff])


def arc_rows(split: str):
    """Yield {'question', 'correct', 'wrong', 'idx'} from ARC Easy+Challenge.

    Same concatenation order, split pattern, and seeded distractor choice as
    sycophancy._factual_rows, so the train/test boundary and the chosen wrong
    answer agree across deception types. Requires network / HF cache.
    """
    from datasets import load_dataset

    raw = []
    for config in _ARC_CONFIGS:
        for hf_split in ("train", "validation", "test"):
            raw.extend(load_dataset("allenai/ai2_arc", config, split=hf_split))

    test_idx = _test_indices(len(raw))
    for i, row in enumerate(raw):
        if (split == "test") != (i in test_idx):
            continue
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if row["answerKey"] not in labels:
            continue  # malformed upstream row
        correct = texts[labels.index(row["answerKey"])]
        wrongs = [t for l, t in zip(labels, texts) if l != row["answerKey"]]
        if not wrongs:
            continue
        wrong = random.Random(f"{_SPLIT_SEED}:distractor:{i}").choice(wrongs)
        yield {"question": row["question"], "correct": correct,
               "wrong": wrong, "idx": i}


def fixture_rows(split: str):
    """Yield the same row schema from fixtures/factual_smoke.jsonl -- the tiny,
    checked-in, OFFLINE stand-in (no network; smoke tests and demos only, NOT
    trustworthy AUROC). Same 95/5 split pattern."""
    path = _FIXTURES_DIR / "factual_smoke.jsonl"
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip()]

    test_idx = _test_indices(len(raw))
    for i, row in enumerate(raw):
        if (split == "test") != (i in test_idx):
            continue
        yield {"question": row["question"], "correct": row["correct"],
               "wrong": row["wrong"], "idx": i}


def rows(source: str, split: str):
    """Dispatch: 'factual' = ARC (network), 'factual-small' = offline fixture."""
    if source == "factual":
        return arc_rows(split)
    if source == "factual-small":
        return fixture_rows(split)
    raise ValueError(f"unknown MCQ source {source!r}; have 'factual', 'factual-small'")
