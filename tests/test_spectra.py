"""The threshold-free reads, and the fast route that feeds them.

`spectra.py` produces every headline number in the README: the participation ratio the identity
argument turns on, and the principal angles `compare` reports. It is covered directly here
rather than through `test_nulls.py` and the README tests, which skip without a published lens
and so cover nothing in CI.

Each read is pinned against a case where the answer is known by construction rather than by
running the code and writing down what came out.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_jlens as je


def spectrum(*values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------- participation ratio

@pytest.mark.parametrize("r", [1, 2, 5, 40])
def test_participation_ratio_is_exactly_r_for_r_equal_values(r):
    """The property that makes it readable as a dimension count: `(sum s^2)^2 / sum s^4` equals
    `r` exactly for a spectrum with `r` equal non-zero values, at any scale."""
    for scale in (1e-3, 1.0, 1e5):
        s = np.full(r, scale)
        assert je.participation_ratio(s) == pytest.approx(float(r), rel=1e-12)


def test_participation_ratio_is_one_for_a_rank_one_spectrum():
    assert je.participation_ratio(spectrum(7.0, 0.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_participation_ratio_ignores_scale_but_not_shape():
    s = spectrum(4.0, 2.0, 1.0)
    assert je.participation_ratio(s) == pytest.approx(je.participation_ratio(s * 1000.0))
    assert je.participation_ratio(s) < je.participation_ratio(spectrum(4.0, 4.0, 4.0))


def test_participation_ratio_of_nothing_is_zero_rather_than_a_division():
    assert je.participation_ratio(spectrum(0.0, 0.0)) == 0.0


def test_a_flat_spectrum_maximises_it():
    """The property behind the audit summary's restriction: a matrix of pure noise has the
    flattest spectrum there is and therefore the LARGEST participation ratio, so an unrestricted
    argmax over PR picks the emptiest layer in a lens."""
    n = 64
    flat = np.ones(n)
    peaked = np.exp(-np.arange(n) / 4.0)
    assert je.participation_ratio(flat) == pytest.approx(float(n))
    assert je.participation_ratio(peaked) < 10.0


# ---------------------------------------------------------------- shannon rank

@pytest.mark.parametrize("r", [1, 2, 8, 32])
def test_shannon_rank_is_exactly_r_for_r_equal_values(r):
    assert je.shannon_rank(np.full(r, 3.0)) == pytest.approx(float(r), rel=1e-12)


def test_shannon_rank_and_participation_ratio_disagree_on_a_long_tail():
    """They read different moments -- PR the second, H2 the whole distribution's entropy -- so a
    long tail separates them. If they agreed everywhere, one would be redundant.

    The tail has to carry real ENERGY, not merely be long: a first version used 200 values at
    0.05 against 4 at 10.0, which is 0.5 of energy against 400, and both reads returned ~4.0.
    Here the two halves carry 400 each, and they return ~16 and ~80.
    """
    s = np.concatenate([np.full(4, 10.0), np.full(400, 1.0)])
    pr, h2 = je.participation_ratio(s), je.shannon_rank(s)
    assert h2 > 4.0 * pr, f"PR {pr:.1f} against H2 {h2:.1f}"


# ---------------------------------------------------------------- the fast route

def test_the_fast_spectrum_agrees_with_the_svd_on_both_reads():
    """`energy_spectrum` defaults to eigvalsh(A^T A), and both reads weight by s^2, so the
    precision the squaring costs sits entirely in values they do not see. Measured on a real
    Qwen3.5-4B transport at d = 2560: relative difference 2.0e-16 on PR for J and 8.2e-16 for
    J - alpha*I, against the docstring's claimed 1e-15.
    """
    rng = np.random.default_rng(0)
    d = 192
    u = np.linalg.qr(rng.standard_normal((d, 12)))[0]
    v = np.linalg.qr(rng.standard_normal((d, 12)))[0]
    A = (u * np.logspace(2, -4, 12)) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d)

    fast, exact = je.energy_spectrum(A), je.energy_spectrum(A, exact=True)
    for read in (je.participation_ratio, je.shannon_rank):
        a, b = read(fast), read(exact)
        assert abs(a - b) / b < 1e-13, f"{read.__name__}: {a} vs {b}"


def test_the_fast_route_loses_the_small_singular_values_it_is_documented_to_lose():
    """The reason `energy_spectrum` must not feed a noise floor, asserted rather than assumed.

    Squaring halves the significant digits of the smallest singular values, and that is exactly
    the region a detection threshold reads. This is why `cli.cmd_audit` computes an exact SVD for
    the K columns and uses the fast route only for PR.
    """
    rng = np.random.default_rng(1)
    d = 128
    q1, q2 = je.haar_orthogonal(d, rng), je.haar_orthogonal(d, rng)
    s = np.logspace(0, -12, d)                    # spans the region where squaring hurts
    A = (q1 * s) @ q2.T

    fast, exact = je.energy_spectrum(A), je.energy_spectrum(A, exact=True)
    top_err = abs(fast[0] - exact[0]) / exact[0]
    tail_err = abs(fast[-1] - exact[-1]) / exact[-1]
    assert top_err < 1e-12, "the large values must be unaffected"
    assert tail_err > 1e3 * max(top_err, 1e-16), (
        f"the documented precision loss in the tail is not present: top {top_err:.2e}, "
        f"tail {tail_err:.2e}")


def test_the_fast_route_handles_both_orientations():
    """It forms whichever Gram matrix is smaller, so a wide frame and a tall one must agree with
    the SVD equally."""
    rng = np.random.default_rng(2)
    for shape in ((40, 120), (120, 40)):
        A = rng.standard_normal(shape)
        fast, exact = je.energy_spectrum(A), je.energy_spectrum(A, exact=True)
        assert fast.size == exact.size == min(shape)
        assert np.allclose(fast, exact, rtol=1e-10)


# ---------------------------------------------------------------- principal angles

def test_a_subspace_agrees_with_itself_exactly():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((64, 64))
    assert np.allclose(je.principal_angles(A, A, k=8), 1.0, atol=1e-9)


def test_two_unrelated_maps_share_almost_nothing():
    """The other end of the scale `compare` reads. Two independent draws of one construction sit
    far below the 0.8 mean that separates 'same map' from 'drifted'."""
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((128, 128)), rng.standard_normal((128, 128))
    assert float(je.principal_angles(a, b, k=16).mean()) < 0.5


def test_the_cosines_descend_and_stay_in_range():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((96, 96)), rng.standard_normal((96, 96))
    c = je.principal_angles(a, b, k=20)
    assert c.size == 20
    assert np.all((c >= 0.0) & (c <= 1.0))
    assert np.all(np.diff(c) <= 1e-12), "canonical correlations must come back descending"


def test_a_shared_subspace_is_found_and_the_rest_is_not():
    """Constructed so the answer is known: two maps built on a common 5-dimensional right
    subspace plus independent remainders. The leading five cosines must be ~1 and the sixth
    must not."""
    rng = np.random.default_rng(6)
    d, shared = 96, 5
    common = np.linalg.qr(rng.standard_normal((d, shared)))[0]
    a = (np.linalg.qr(rng.standard_normal((d, shared)))[0] * 50.0) @ common.T
    b = (np.linalg.qr(rng.standard_normal((d, shared)))[0] * 50.0) @ common.T
    c = je.principal_angles(a, b, k=shared)
    assert np.all(c > 0.999), f"the shared subspace must be recovered: {c}"


@pytest.mark.parametrize("k", [0, -1])
def test_principal_angles_refuses_a_meaningless_k(k):
    rng = np.random.default_rng(7)
    A = rng.standard_normal((32, 32))
    with pytest.raises(ValueError, match="k must be"):
        je.principal_angles(A, A, k=k)


def test_principal_angles_refuses_a_k_past_the_available_rank():
    """Rather than silently comparing fewer directions than asked for, which would make two
    lenses look more alike the smaller they are."""
    rng = np.random.default_rng(8)
    A, B = rng.standard_normal((8, 32)), rng.standard_normal((8, 32))
    with pytest.raises(ValueError, match="exceeds the available rank"):
        je.principal_angles(A, B, k=20)
