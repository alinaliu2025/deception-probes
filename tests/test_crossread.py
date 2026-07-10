"""Fast checks for scripts/crossread.py. No model download.

The heavy imports (torch/transformers) live inside crossread.main(), so the
module itself imports on numpy+sklearn+datasets only.
"""

import numpy as np
import pytest

from dprobe.data import sycophancy
from scripts.crossread import commit_twin, render_table, sweep_2x2


def _text_child(label: int, sampled: str):
    from dprobe.data.base import Example
    return Example("sys", "PRESSURE Q body", label, "sycophancy",
                   meta={"matching": " (B)", "not_matching": " (A)",
                         "neutral_user": "Q body", "design": "rollout",
                         "assistant_prefix": sampled})


def test_commit_twin_swaps_prefix_only():
    """Twin = same example, prefix -> bare answer tag of its own label."""
    caved = _text_child(1, "Yes, you're right, it's (B)")
    held = _text_child(0, "Actually I believe (A)")
    tc, th = commit_twin(caved), commit_twin(held)
    assert tc.meta["assistant_prefix"] == " (B)"   # label 1 -> matching tag
    assert th.meta["assistant_prefix"] == " (A)"   # label 0 -> not_matching tag
    for orig, twin in ((caved, tc), (held, th)):
        assert twin.user == orig.user and twin.label == orig.label
        assert twin.system == orig.system
        rest_o = {k: v for k, v in orig.meta.items() if k != "assistant_prefix"}
        rest_t = {k: v for k, v in twin.meta.items() if k != "assistant_prefix"}
        assert rest_o == rest_t
    # originals untouched (replace, not mutation)
    assert caved.meta["assistant_prefix"].startswith("Yes")


def test_sweep_2x2_shared_signal_transfers():
    """Synthetic twins where BOTH reads carry the same planted direction:
    all four cells must be high. Then break the text read's signal: the
    diagonals split and the cross cells collapse."""
    rng = np.random.default_rng(0)
    n, layers, hidden = 240, 3, 16
    y = np.array([0, 1] * (n // 2))
    groups = [f"q{i // 2}" for i in range(n)]  # pair rows like real twins

    base = rng.normal(size=(n, layers, hidden))
    shared = base.copy()
    shared[y == 1, :, 0] += 4.0
    acts_c, acts_t = shared, shared + rng.normal(scale=0.1, size=shared.shape)

    curves, (tr, te) = sweep_2x2(acts_c, acts_t, y, groups, "mms", "test")
    for cell in ("cc", "ct", "tt", "tc"):
        assert curves[cell].max() > 0.9, cell
    assert set(np.array(groups)[tr]).isdisjoint(set(np.array(groups)[te]))

    # now give the text read NO signal: its diagonal and both cross cells die
    acts_t_dead = rng.normal(size=shared.shape)
    curves2, _ = sweep_2x2(acts_c, acts_t_dead, y, groups, "mms", "test")
    assert curves2["cc"].max() > 0.9
    assert curves2["tt"].max() < 0.8
    assert curves2["ct"].max() < 0.8 and curves2["tc"].max() < 0.8


def test_render_table_has_all_cells():
    curves = {k: np.array([0.5, 0.9]) for k in ("cc", "ct", "tt", "tc")}
    table = render_table(curves)
    assert "C->C" in table and "T->C" in table
    assert table.count("\n") == 2  # header + 2 layers


def test_crossread_uses_text_children_from_real_build():
    """End-to-end twin derivation on real (offline fixture) rollout examples:
    parents from the factual-small build produce twins whose prefixes are the
    bare tags, matching the parent's own meta."""
    ex = sycophancy.build(design="rollout", source="factual-small")[0]
    for label, expect_key in ((1, "matching"), (0, "not_matching")):
        from dataclasses import replace
        child = replace(ex, label=label,
                        meta={**ex.meta, "assistant_prefix": "blah blah (X)"})
        twin = commit_twin(child)
        assert twin.meta["assistant_prefix"] == ex.meta[expect_key]
