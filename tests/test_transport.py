"""The resolved rank of a transport, against planted truth."""
import numpy as np
import pytest

import entroptics_jlens as je
from entroptics_jlens.frames import FrameError


def planted(d, K, top, low, rng, noise=1.0):
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    s = np.linspace(top, low, K)
    return (U * s) @ V.T + noise * rng.standard_normal((d, d))


@pytest.mark.parametrize("top,low", [(120.0, 60.0), (60.0, 40.0), (40.0, 34.0)])
def test_planted_rank_is_recovered_exactly(top, low):
    """Every planted mode above the BBP threshold is resolved, and no extra one is.

    The bulk edge of a d x d unit-variance matrix sits at ~2*sqrt(d) = 32 for d = 256, and an
    additive spike of strength s is observed at ~s + d/s, so a planted mode is detectable only
    above that threshold. All three bands here are supra-threshold; sub-threshold planting is
    the separate case in ``test_subthreshold_planting_is_not_resolved``."""
    rng = np.random.default_rng(0)
    spec = je.transport_spectrum(planted(256, 12, top, low, rng))
    assert spec.K == 12
    assert not spec.saturated


def test_subthreshold_planting_is_not_resolved():
    """Modes below the bulk edge are not reported. This is the instrument being right, not
    failing: a spike buried in the bulk carries no evidence against the null."""
    rng = np.random.default_rng(0)
    spec = je.transport_spectrum(planted(256, 12, 20.0, 8.0, rng))
    assert spec.K < 12


def test_recount_is_monotone_in_far():
    """A different false-alarm level is a re-count of the same evidence, never a re-measurement."""
    rng = np.random.default_rng(1)
    spec = je.transport_spectrum(planted(192, 8, 90.0, 50.0, rng))
    counts = [spec.recount(f) for f in (1e-6, 1e-3, 0.05, 0.2)]
    assert counts == sorted(counts)
    assert spec.recount(spec.far) == spec.K


def test_resolved_transport_has_exact_rank_K():
    rng = np.random.default_rng(2)
    J = planted(192, 8, 90.0, 50.0, rng)
    J_K, spec = je.resolved_transport(J)
    assert np.linalg.matrix_rank(J_K, tol=1e-8 * np.linalg.norm(J_K, 2)) == spec.K
    assert J_K.shape == J.shape


def test_pure_noise_transport_refuses_rather_than_returning_J_whole():
    """A transport that resolves nothing has no rank to truncate at. Returning J unchanged
    would report a full-rank map as a resolved one."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="resolves 0 modes"):
        je.resolved_transport(rng.standard_normal((128, 128)))


def test_float16_is_upcast_not_read_in_place():
    """Lens checkpoints ship float16; every read here is spectral and needs float64."""
    rng = np.random.default_rng(4)
    J = planted(192, 8, 90.0, 50.0, rng)
    spec16 = je.transport_spectrum(np.asarray(J, dtype=np.float16))
    assert spec16.singular.dtype == np.float64
    assert spec16.K == je.transport_spectrum(J).K


@pytest.mark.parametrize("bad,match", [
    (np.zeros((4, 4, 4)), "2-D"),
    (np.zeros((0, 4)), "non-empty"),
    (np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
])
def test_broken_surfaces_refuse(bad, match):
    with pytest.raises(FrameError, match=match):
        je.as_frame(bad, name="J")


def test_far_reaches_the_mp_provider_and_not_the_robust_fence():
    """`robust` is `null_providers.robust`, the Tukey fence Q3 + 1.5*IQR of the SPECTRUM. It is
    not a calibrated null and takes no false-alarm rate, so `far` is a silent no-op on it.

    Pinned because the two providers sit in adjacent columns of `entroptics-jlens audit` under
    one `--far`, and `TransportSpectrum.far` records the value for both. Measured on gpt2's
    layer-5 identity-free transport: mp's floor moves 2.136 -> 2.196 across far 0.5 -> 1e-6
    while robust holds 2.40679 at every value.
    """
    rng = np.random.default_rng(11)
    d, k = 128, 10
    u = np.linalg.qr(rng.standard_normal((d, k)))[0]
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    A = (u * np.linspace(12.0, 4.0, k)) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d)
    s = je.transport_spectrum(A).singular

    mp = [je.transport_spectrum(A, far=f, null="mp", s=s).floor for f in (0.5, 0.05, 5e-4, 1e-6)]
    rob = [je.transport_spectrum(A, far=f, null="robust", s=s).floor
           for f in (0.5, 0.05, 5e-4, 1e-6)]

    assert len(set(rob)) == 1, f"robust floor moved with far: {rob}"
    assert mp[0] < mp[-1], f"mp floor must tighten as far shrinks: {mp}"
    # The recorded `far` is the caller's, on both -- which is why the docstring says to read it
    # together with null_name rather than as a property of the number.
    assert je.transport_spectrum(A, far=1e-6, null="robust", s=s).far == 1e-6
