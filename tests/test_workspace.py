"""The runtime half: build once, read per token.

The properties that make this usable in a serving path are exactness and isolation, not speed --
speed is a consequence of the SVD being preprocessing. So these pin what `inject` moves and what
it leaves alone, which is the part a caller has to be able to rely on.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_jlens as je


def transport(d=96, k=10, alpha=2.0, seed=0):
    """A transport shaped like a real one: a low-rank map plus noise plus an identity core."""
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.standard_normal((d, k)))[0]
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    return (u * np.linspace(9.0, 4.0, k)) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d) \
        + alpha * np.eye(d)


def test_the_basis_is_orthonormal():
    """Everything below depends on it: injection isolation, the exact round trip, and reading a
    coordinate as a projection rather than as a regression."""
    ws = je.workspace(transport(), layer=6)
    gram = ws.basis.T @ ws.basis
    assert np.allclose(gram, np.eye(ws.k), atol=1e-10)
    assert np.allclose(ws.image.T @ ws.image, np.eye(ws.k), atol=1e-10)


def test_it_is_built_on_the_identity_free_transport():
    """On raw J the leading directions at depth are the identity's, so a coordinate would be
    reading the residual stream passing through rather than the map."""
    J = transport(alpha=6.0)
    ws = je.workspace(J, layer=6)
    assert ws.identity_energy > 0.5, "the fixture must be identity-dominated to test this"
    m = je.decompose(J).residual
    direct = np.linalg.svd(m, full_matrices=False)[2][: ws.k].T
    assert np.allclose(np.abs(ws.basis), np.abs(direct), atol=1e-9)


def test_inject_moves_one_coordinate_and_leaves_the_others_where_they_were():
    """The property that makes it a controlled write rather than a perturbation. Machine
    precision, not approximately: the basis is orthonormal, so the other k-1 coordinates are
    untouched exactly."""
    ws = je.workspace(transport(), layer=6)
    h = np.random.default_rng(1).standard_normal(ws.d)
    for j in (0, ws.k // 2, ws.k - 1):
        moved = ws.extract(ws.inject(h, j, 5.0)) - ws.extract(h)
        assert moved[j] == pytest.approx(5.0, abs=1e-9)
        others = np.delete(moved, j)
        assert np.abs(others).max() < 1e-12, f"direction {j} disturbed others by {np.abs(others).max()}"


def test_inject_does_not_modify_the_caller_s_array():
    """An in-place write into a model's activation is easy to leave switched on."""
    ws = je.workspace(transport(), layer=6)
    h = np.random.default_rng(2).standard_normal(ws.d)
    before = h.copy()
    ws.inject(h, 0, 100.0)
    assert np.array_equal(h, before)


def test_extract_and_reconstruct_round_trip_exactly():
    ws = je.workspace(transport(), layer=6)
    coords = np.random.default_rng(3).standard_normal((7, ws.k))
    assert np.allclose(ws.extract(ws.reconstruct(coords)), coords, atol=1e-10)


def test_extract_takes_one_token_or_a_batch():
    ws = je.workspace(transport(), layer=6)
    rng = np.random.default_rng(4)
    assert ws.extract(rng.standard_normal(ws.d)).shape == (ws.k,)
    assert ws.extract(rng.standard_normal((32, ws.d))).shape == (32, ws.k)


def test_a_fixed_width_can_be_forced():
    """A serving system usually wants a stable coordinate shape across model versions."""
    ws = je.workspace(transport(), layer=6, k=4)
    assert ws.k == 4 and ws.basis.shape == (ws.d, 4)


def test_the_wrong_width_residual_is_refused():
    ws = je.workspace(transport(d=96), layer=6)
    with pytest.raises(ValueError, match="on one basis"):
        ws.extract(np.zeros((4, 64)))
    with pytest.raises(ValueError, match="wide"):
        ws.inject(np.zeros(64), 0, 1.0)


def test_a_direction_outside_the_workspace_is_refused():
    ws = je.workspace(transport(), layer=6)
    for bad in (-1, ws.k, ws.k + 100):
        with pytest.raises(ValueError, match="outside"):
            ws.inject(np.zeros(ws.d), bad, 1.0)


def test_a_transport_resolving_nothing_refuses_rather_than_returning_an_empty_workspace():
    """An empty coordinate vector would flow through a pipeline as a valid reading of nothing."""
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((64, 64)) / 8.0
    with pytest.raises(ValueError, match="no workspace to read"):
        je.workspace(noise, layer=0)


def test_the_directions_come_back_ordered_by_strength():
    """`inject(h, 0, ...)` has to mean the strongest direction, or a caller reading the docstring
    moves the wrong thing."""
    ws = je.workspace(transport(), layer=6)
    assert np.all(np.diff(ws.singular) <= 1e-9)
