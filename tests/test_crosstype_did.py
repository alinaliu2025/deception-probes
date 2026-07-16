"""Fast checks for the DiD cross-type transfer path (scripts/crosstype_did.py +
the transfer_matrix/direction_cosines primitives). Pure numpy -- no model, no
network. Pins the two things a silent bug would break: (1) each probe is read at
its OWN layer out of the target dataset's arrows, (2) the probe-spec parser.
"""

import numpy as np

from dprobe.evaluate import direction_cosines, transfer_matrix
from dprobe.probes import Probe


def _probe(direction, layer, name):
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-8)
    return Probe(d, 0.0, layer, "mms", name)


def test_transfer_matrix_reads_each_probe_at_its_own_layer():
    """A probe trained at layer L must be scored at layer L of the target arrows,
    even when probes in the matrix live at different layers."""
    H = 4
    # dataset A: separable ONLY at layer 1; dataset B: separable ONLY at layer 2
    labels = np.array([0, 0, 1, 1])
    arrows_A = np.zeros((4, 3, H)); arrows_A[labels == 1, 1, 0] = 5.0
    arrows_B = np.zeros((4, 3, H)); arrows_B[labels == 1, 2, 1] = 5.0

    pA = _probe([1, 0, 0, 0], layer=1, name="A")   # points at the L1 signal
    pB = _probe([0, 1, 0, 0], layer=2, name="B")   # points at the L2 signal
    probes = {"A": pA, "B": pB}
    acts = {"A": arrows_A, "B": arrows_B}
    labs = {"A": labels, "B": labels}

    M, order = transfer_matrix(probes, acts, labs)
    i, j = order.index("A"), order.index("B")
    # diagonal: each probe perfectly separates its own dataset at its own layer
    assert M[i, i] == 1.0 and M[j, j] == 1.0
    # off-diagonal probe A on dataset B: A reads layer 1 of B, where B has NO
    # signal (B's signal is at layer 2) -> chance. Proves the per-layer readout.
    assert abs(M[i, j] - 0.5) < 1e-9


def test_direction_cosines_symmetric_unit_diagonal():
    p1 = _probe([1, 0, 0, 0], 1, "x")
    p2 = _probe([1, 1, 0, 0], 1, "y")
    C, order = direction_cosines({"x": p1, "y": p2})
    assert np.allclose(np.diag(C), 1.0)
    assert np.allclose(C, C.T)
    assert 0.0 < C[0, 1] < 1.0


def test_parse_spec_variants():
    from scripts.crosstype_did import parse_spec
    assert parse_spec("syco=path/to/probe.npz") == (
        "syco", "path/to/probe.npz", "instructed")
    assert parse_spec("sand-ince=r/probe.npz@incentive") == (
        "sand-ince", "r/probe.npz", "incentive")
    import pytest
    with pytest.raises(Exception):
        parse_spec("no_equals_sign")
