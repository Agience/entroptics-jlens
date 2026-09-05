"""The nulls. A read without one is a fit."""
import numpy as np
import pytest

import entroptics_jlens as je


@pytest.mark.parametrize("seed", range(6))
def test_gaussian_null_reads_no_structure(seed):
    """Floor calibration, under every provider. If a derived edge resolved modes in pure noise,
    nothing downstream would mean anything -- and the robust provider, which exists to stop the
    signal lifting its own floor, must not buy that by manufacturing modes."""
    rng = np.random.default_rng(seed)
    G = je.gaussian_null((256, 256), rng)
    for name, spec in je.spectrum_under_nulls(G).items():
        assert spec.K == 0, f"{name} resolved {spec.K} modes in pure noise at seed {seed}"


@pytest.mark.parametrize("sigma", [1e-3, 1.0, 1e3])
def test_resolved_count_is_scale_invariant(sigma):
    """The floor estimates its per-cell variance from the screen it is handed, so it rescales
    with the matrix."""
    rng = np.random.default_rng(7)
    U = je.haar_orthogonal(192, rng)[:, :6]
    V = je.haar_orthogonal(192, rng)[:, :6]
    J = (U * np.linspace(90.0, 60.0, 6)) @ V.T + rng.standard_normal((192, 192))
    assert je.transport_spectrum(sigma * J).K == je.transport_spectrum(J).K


def test_shuffling_entries_destroys_the_resolved_rank():
    """The strongest 'is anything here' control: the entry multiset is preserved exactly, so
    no distributional assumption is doing the work."""
    rng = np.random.default_rng(8)
    U = je.haar_orthogonal(256, rng)[:, :10]
    V = je.haar_orthogonal(256, rng)[:, :10]
    J = (U * np.linspace(120.0, 70.0, 10)) @ V.T + rng.standard_normal((256, 256))
    assert je.transport_spectrum(J).K == 10
    assert je.transport_spectrum(je.shuffled_entries(J, rng)).K == 0


def test_matched_spectrum_preserves_the_spectrum_exactly():
    """Which is the point: it is the control for reads taken THROUGH J, not for K."""
    rng = np.random.default_rng(9)
    U = je.haar_orthogonal(128, rng)[:, :5]
    V = je.haar_orthogonal(128, rng)[:, :5]
    J = (U * np.linspace(80.0, 50.0, 5)) @ V.T + rng.standard_normal((128, 128))
    s_J = np.linalg.svd(J, compute_uv=False)
    s_M = np.linalg.svd(je.matched_spectrum(J, rng), compute_uv=False)
    assert np.allclose(s_J, s_M, rtol=1e-10, atol=1e-10)


def test_haar_orthogonal_is_orthogonal():
    rng = np.random.default_rng(10)
    Q = je.haar_orthogonal(64, rng)
    assert np.allclose(Q.T @ Q, np.eye(64), atol=1e-12)


def test_frobenius_sigma_matches_a_known_scale():
    rng = np.random.default_rng(11)
    assert je.frobenius_sigma(3.0 * rng.standard_normal((400, 400))) == pytest.approx(3.0, rel=0.02)


def test_the_default_null_under_counts_when_the_signal_lifts_its_own_floor():
    """The mp provider estimates its per-cell variance from the whole matrix, signal included,
    so a transport carrying a lot of structure hides its own weakest modes. Every one of these
    30 planted modes is supra-threshold by the BBP criterion (s + d/s > 2*sqrt(d)); mp resolves
    19, robust resolves 29. K is a LOWER BOUND on the resolved rank, and the bound is loosest
    exactly where the answer matters."""
    rng = np.random.default_rng(0)
    d, K = 256, 30
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    s = np.linspace(400.0, 40.0, K)
    J = (U * s) @ V.T + rng.standard_normal((d, d))
    assert int(((s + d / s) > 2 * np.sqrt(d)).sum()) == K       # all detectable in principle

    under = je.spectrum_under_nulls(J)
    assert under["mp"].K == 19
    assert under["robust"].K == 29
    assert under["mp"].K < under["robust"].K <= K


def test_a_narrow_planted_rank_is_unaffected_by_the_choice_of_null():
    """The bias grows with the number of planted modes and their spread; a small, tight rank
    reads the same under both providers."""
    rng = np.random.default_rng(1)
    d, K = 256, 10
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    J = (U * np.linspace(120.0, 40.0, K)) @ V.T + rng.standard_normal((d, d))
    under = je.spectrum_under_nulls(J)
    assert under["mp"].K == under["robust"].K == K


def test_both_nulls_read_no_structure_in_pure_noise():
    """The bias fix must not cost calibration: robust may not manufacture modes."""
    rng = np.random.default_rng(2)
    for name, spec in je.spectrum_under_nulls(je.gaussian_null((256, 256), rng)).items():
        assert spec.K == 0, name


def test_an_unknown_null_refuses_rather_than_falling_back():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="unknown null"):
        je.transport_spectrum(je.gaussian_null((64, 64), rng), null="whatever")


def test_a_heavy_tailed_entry_distribution_defeats_the_gaussian_edge():
    """The floor is the edge of an i.i.d. GAUSSIAN ensemble. An i.i.d. matrix whose entries are
    heavy-tailed puts singular values above that edge with no low-rank structure at all, so on
    such a matrix K must be read against the shuffled-entry baseline and never against 0.

    Real Jacobian transports are heavy-tailed (gpt2 layer 10: excess kurtosis 7.4e4, max entry
    455x the rms), which is why the shuffled control fires on them."""
    rng = np.random.default_rng(0)
    d = 256
    heavy = rng.standard_normal((d, d)) * rng.standard_exponential((d, d)) ** 3   # iid, no rank
    spec = je.transport_spectrum(heavy)
    assert spec.excess_kurtosis > 50
    assert spec.K > 0, "an iid heavy-tailed matrix should breach the Gaussian edge"
    assert je.transport_spectrum(je.shuffled_entries(heavy, rng)).K > 0

    gauss = je.gaussian_null((d, d), rng)
    assert abs(je.transport_spectrum(gauss).excess_kurtosis) < 1.0
    assert je.transport_spectrum(gauss).K == 0


def test_an_entry_shuffle_destroys_an_identity_core_and_inverts_the_comparison():
    """The entry permutation moves the diagonal off the diagonal, so it is not a null for a
    matrix the architecture guarantees has an identity component.

    The identity raises the real matrix's self-estimated floor and the shuffle removes it from
    the control's, so the control resolves MORE. Measured on gpt2 layer 9 (mp, far=0.05): raw J
    reads 6 against its shuffle's 34; J - alpha*I reads 46 against 21, stable over 8 seeds.
    Reproduced here on a constructed transport of the same shape.
    """
    rng = np.random.default_rng(5)
    d, k = 192, 12
    u = np.linalg.qr(rng.standard_normal((d, k)))[0]
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    M = (u * 4.0) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d)
    J = M + 3.0 * np.eye(d)                       # the identity a residual stream adds

    k_raw = je.transport_spectrum(J, null="mp").K
    k_raw_shuf = je.transport_spectrum(je.shuffled_entries(J, np.random.default_rng(0)),
                                       null="mp").K
    residual = je.decompose(J).residual
    k_dec = je.transport_spectrum(residual, null="mp").K
    k_dec_shuf = je.transport_spectrum(je.shuffled_entries(residual, np.random.default_rng(0)),
                                       null="mp").K

    assert k_raw < k_raw_shuf, (
        f"premise broken: raw J ({k_raw}) should read below its own shuffle ({k_raw_shuf}) when "
        f"an identity core is present, which is the whole point of this test")
    assert k_dec > k_dec_shuf, (
        f"after decompose the real transport ({k_dec}) must out-resolve its shuffle "
        f"({k_dec_shuf})")
