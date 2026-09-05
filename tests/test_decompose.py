"""Removing the architectural component before reading the transport."""
import numpy as np
import pytest

import entroptics_jlens as je


def test_alpha_is_the_exact_frobenius_projection_onto_the_identity():
    """alpha = tr(J)/d = <J,I>/<I,I>. Not a fit: the unique least-squares coefficient against a
    basis element the residual stream guarantees is present."""
    rng = np.random.default_rng(0)
    d = 64
    J = rng.standard_normal((d, d)) + 0.7 * np.eye(d)
    dec = je.decompose(J, kind="identity")
    assert dec.alpha == pytest.approx(np.trace(J) / d, rel=1e-12)
    # the residual is orthogonal to the identity: <J - aI, I> = 0
    assert float(np.trace(dec.residual)) == pytest.approx(0.0, abs=1e-9 * np.linalg.norm(J))
    # and alpha minimises ||J - c I||_F over c
    for c in (dec.alpha * 0.5, dec.alpha * 1.5, dec.alpha + 0.3):
        assert np.linalg.norm(J - c * np.eye(d)) >= np.linalg.norm(dec.residual) - 1e-9


def test_identity_energy_is_recovered():
    rng = np.random.default_rng(1)
    d = 128
    M = rng.standard_normal((d, d)) * 0.01
    J = M + 1.0 * np.eye(d)
    dec = je.decompose(J, kind="identity")
    assert dec.removed_energy > 0.9                      # overwhelmingly identity
    assert dec.identity_dominated
    assert np.allclose(dec.residual, M - np.diag(np.diag(M)) + np.diag(np.diag(M))
                       - (dec.alpha - 1.0) * np.eye(d), atol=1e-9)


def test_removing_the_identity_uncovers_modes_it_was_burying():
    """An identity block is a FLAT spectral component. It inflates the estimated per-cell
    variance, lifts the floor, and buries real structure underneath -- which is exactly what a
    deep transport does (Qwen layer 30: 79% identity energy, K 25 -> 183 once removed)."""
    rng = np.random.default_rng(2)
    d, K = 256, 8
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    M = (U * np.linspace(70.0, 45.0, K)) @ V.T + rng.standard_normal((d, d))
    J = M + 40.0 * np.eye(d)                             # a large flat block on top
    assert je.transport_spectrum(J).K < K
    assert je.transport_spectrum(je.decompose(J, kind="identity").residual).K == K


def test_diagonal_decomposition_removes_d_coefficients():
    rng = np.random.default_rng(3)
    d = 64
    J = rng.standard_normal((d, d))
    dec = je.decompose(J, kind="diagonal")
    assert np.allclose(np.diag(dec.residual), 0.0)
    assert np.isnan(dec.alpha)


@pytest.mark.parametrize("bad,match", [
    ("rect", "square"),
    ("kind", "unknown decomposition"),
])
def test_decompose_refuses(bad, match):
    rng = np.random.default_rng(4)
    if bad == "rect":
        with pytest.raises(ValueError, match=match):
            je.decompose(rng.standard_normal((32, 16)))
    else:
        with pytest.raises(ValueError, match=match):
            je.decompose(rng.standard_normal((16, 16)), kind="whatever")
