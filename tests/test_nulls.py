"""Distribution-free floors: the surrogates, the fast scorer, and the calibration check."""
import numpy as np
import pytest

import entroptics_jlens as je
from entroptics.null_providers import top_spectrum_value


@pytest.mark.parametrize("shape", [(64, 64), (200, 80), (300, 300)])
def test_fast_scorer_matches_the_library_exactly(shape):
    """A sampled floor needs draws >> 1/far scores and a full SVD at d=2560 is seconds apiece.
    The fast scorer must be the same number, not an approximation of it."""
    rng = np.random.default_rng(0)
    A = (rng.standard_normal(shape)
         + (rng.standard_normal((shape[0], 3)) * 8) @ rng.standard_normal((3, shape[1])))
    fast = je.top_singular(A, rng=np.random.default_rng(1))
    assert fast == pytest.approx(top_spectrum_value(A, "projection"), rel=1e-12)


def test_sign_flip_preserves_the_magnitude_profile_exactly():
    """|J| entrywise, hence every row energy, every column energy and the Frobenius norm.
    Only the sign coherence dies -- the tightest entropy-matched null for a dense operator."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((128, 96)) * rng.standard_exponential((128, 96)) ** 2
    S = je.sign_flip(A, rng)
    assert np.array_equal(np.abs(S), np.abs(A))
    assert np.array_equal(np.sort((S ** 2).sum(1)), np.sort((A ** 2).sum(1)))


@pytest.mark.parametrize("fn,axis", [(je.within_row_shuffle, 1), (je.within_col_shuffle, 0)])
def test_within_shuffles_preserve_their_multisets(fn, axis):
    """A permutation moves values without altering them, so each line's multiset survives.
    (The SUM of a reordered multiset differs in last-bit rounding, so compare the multiset.)"""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((64, 48)) * rng.standard_exponential((64, 48)) ** 2
    S = fn(A, rng)
    assert np.array_equal(np.sort(A, axis=axis), np.sort(S, axis=axis))


def test_a_sampled_floor_is_calibrated_where_the_gaussian_edge_is_not():
    """A provider that resolves modes in a draw from its OWN null is not a floor. On a heavy-
    tailed matrix with no rank structure the Gaussian edge fails that check and the sampled
    floor passes it."""
    rng = np.random.default_rng(3)
    d = 192
    heavy = rng.standard_normal((d, d)) * rng.standard_exponential((d, d)) ** 3
    assert je.transport_spectrum(heavy).K > 0                       # mp fires on pure noise
    sf = je.sampled_floor(heavy, surrogate="sign_flip", far=0.05, draws=200, seed=0)
    assert sf.valid, f"exceedance {sf.exceedance} exceeds far {sf.far}"
    assert sf.floor > je.transport_spectrum(heavy).floor


def test_a_sampled_floor_still_finds_planted_structure():
    """Calibration must not be bought with blindness."""
    rng = np.random.default_rng(4)
    d, K = 192, 6
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    J = (U * np.linspace(180.0, 120.0, K)) @ V.T + rng.standard_normal((d, d))
    sf = je.sampled_floor(J, surrogate="sign_flip", far=0.05, draws=200, seed=0)
    assert sf.K == K
    assert sf.valid, f"exceedance {sf.exceedance} exceeds far {sf.far}"


def test_unknown_surrogate_refuses():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="unknown surrogate"):
        je.sampled_floor(rng.standard_normal((32, 32)), surrogate="nope", draws=4)


def test_the_calibration_check_uses_held_out_draws():
    """Scoring against the same sample the quantile came from returns `far` by construction and
    tests nothing. The check draws must be independent of the floor draws."""
    rng = np.random.default_rng(6)
    A = rng.standard_normal((96, 96))
    sf = je.sampled_floor(A, surrogate="sign_flip", far=0.05, draws=200, check_draws=200, seed=0)
    assert sf.check.size == 200
    assert not np.array_equal(np.sort(sf.tops), np.sort(sf.check))
    assert sf.valid


def test_the_empirical_quantile_is_conservative_not_exact_at_few_draws():
    """The (1-far) quantile of a right-skewed null, from a few hundred draws, lands high more
    often than low: the floor is VALID (no excess false alarms) while not being CALIBRATED
    (the rate is below nominal). That costs sensitivity, never validity, and it is why `valid`
    and `calibrated` are separate properties."""
    rng = np.random.default_rng(4)
    d, K = 192, 6
    U = je.haar_orthogonal(d, rng)[:, :K]
    V = je.haar_orthogonal(d, rng)[:, :K]
    J = (U * np.linspace(180.0, 120.0, K)) @ V.T + rng.standard_normal((d, d))
    sf = je.sampled_floor(J, surrogate="sign_flip", far=0.05, draws=200, seed=0)
    assert sf.valid and not sf.calibrated
    assert sf.exceedance < sf.far
    assert sf.K == K                    # conservative, and still finds every planted mode


def test_energy_spectrum_agrees_with_the_svd_where_it_is_used():
    """The eigvalsh route is 3.2x faster at d=2560. PR and H2 weight by s^2, so the precision the
    squaring costs sits entirely in values those reads cannot see -- but that is a claim to check,
    not to assume."""
    rng = np.random.default_rng(0)
    for shape in ((256, 256), (300, 120), (120, 300)):
        A = rng.standard_normal(shape) * rng.standard_exponential(shape)
        fast, exact = je.energy_spectrum(A), je.energy_spectrum(A, exact=True)
        assert je.participation_ratio(fast) == pytest.approx(
            je.participation_ratio(exact), rel=1e-10)
        assert je.shannon_rank(fast) == pytest.approx(je.shannon_rank(exact), rel=1e-10)


def test_participation_ratio_counts_equal_modes_exactly():
    assert je.participation_ratio(np.ones(17)) == pytest.approx(17.0, rel=1e-12)
    assert je.participation_ratio(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0, rel=1e-12)
    assert je.shannon_rank(np.ones(17)) == pytest.approx(17.0, rel=1e-12)


def test_principal_angles_are_one_for_a_shared_subspace_and_fall_apart_for_independent_ones():
    rng = np.random.default_rng(1)
    d, k = 128, 8
    Q = je.haar_orthogonal(d, rng)
    A = (Q[:, :k] * np.linspace(50.0, 30.0, k)) @ Q[:, :k].T
    assert je.principal_angles(A, A, k) == pytest.approx(np.ones(k), abs=1e-9)
    B = (Q[:, k:2 * k] * np.linspace(50.0, 30.0, k)) @ Q[:, k:2 * k].T
    assert je.principal_angles(A, B, k).mean() < 0.2


def test_principal_angles_refuse_a_k_beyond_the_rank():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="exceeds the available rank"):
        je.principal_angles(rng.standard_normal((16, 8)), rng.standard_normal((16, 8)), 12)


def test_every_magnitude_preserving_surrogate_collapses_on_a_single_outlier_cell():
    """The surrogates exist to handle massive-activation coordinates, and they cannot.

    Preserving |J| exactly means the giant cell rides along in every draw, and one giant cell
    carries a large top singular value by itself, so the floor climbs above the real structure.
    Measured on gpt2's identity-free transports, where the largest cell is 70x the rms at layer 0
    and 599x at layer 10: sign_flip reads K = 2, 1, 1, 1 where mp reads 67, 51, 46, 39.

    The clean column is asserted first. Without it this test would pass on a broken sampled
    floor that returned 0 for everything.
    """
    rng = np.random.default_rng(0)
    d, planted = 256, 6
    U = je.haar_orthogonal(d, rng)[:, :planted]
    V = je.haar_orthogonal(d, rng)[:, :planted]
    clean = (U * np.linspace(60.0, 40.0, planted)) @ V.T + rng.standard_normal((d, d))
    spiked = clean.copy()
    spiked[100, 200] = 400.0                      # ~360x the rms of `clean`

    for name in sorted(je.SURROGATES):
        k_clean = je.sampled_floor(clean, surrogate=name, far=0.05, draws=120).K
        k_spiked = je.sampled_floor(spiked, surrogate=name, far=0.05, draws=120).K
        assert k_clean == planted, f"{name} must recover the planted rank cleanly, got {k_clean}"
        assert k_spiked <= 1, f"{name} is expected to collapse on the spike, got {k_spiked}"

    # mp moves the other way on the same matrix: it over-counts rather than collapsing.
    assert je.transport_spectrum(spiked, null="mp").K > planted


def test_sign_flip_is_calibrated_where_it_is_appropriate():
    """The positive control for the test above: on a matrix without a dominating cell the
    sampled floor keeps its promise, so the collapse is about the outlier and not the method."""
    rng = np.random.default_rng(0)
    d, planted = 256, 6
    U = je.haar_orthogonal(d, rng)[:, :planted]
    V = je.haar_orthogonal(d, rng)[:, :planted]
    A = (U * np.linspace(60.0, 40.0, planted)) @ V.T + rng.standard_normal((d, d))
    sf = je.sampled_floor(A, surrogate="sign_flip", far=0.05, draws=200)
    assert sf.K == planted
    assert sf.valid, f"exceedance {sf.exceedance} against far {sf.far}"


def test_calibration_report_scores_the_null_against_the_nulls_own_floor():
    """Calibration is "apply the provider to noise and it finds nothing", so both halves have to
    come from the noise: the null draw's modes are counted against the floor computed on that
    draw, not against the floor computed on the real matrix.

    The two coincide on `mp`, because `sign_flip` preserves |J| entrywise so the per-cell
    variance and hence the MP edge are identical -- asserted below, since a check that cannot
    separate them proves nothing. They differ on `robust`, whose fence is a quantile of the
    spectrum: at gpt2
    layer 0 the old reading gave K_null 117 against the correct 11.
    """
    rng = np.random.default_rng(3)
    d, k = 160, 8
    u = np.linalg.qr(rng.standard_normal((d, k)))[0]
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    A = (u * 9.0) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d)

    rep = je.calibration_report(A, far=0.05, draws=60)

    # mp: the floors must coincide exactly, which is what hid the defect.
    assert rep["mp"]["floor"] == pytest.approx(rep["mp"]["floor_null"], rel=1e-12)

    # robust: they must not, or this fixture cannot distinguish the two readings.
    assert rep["robust"]["floor"] != pytest.approx(rep["robust"]["floor_null"], rel=1e-6), (
        "premise broken: the fence gives the same floor on the matrix and on its sign flip, so "
        "this test cannot tell which floor K_null was counted against")

    # And the reported K_null is the one taken against the null's own floor.
    from entroptics import null_providers
    from entroptics.projection import noise_floor
    draw = je.sign_flip(je.as_frame(A), np.random.default_rng(0))
    sv_null = np.linalg.svd(draw, compute_uv=False)
    for name in ("mp", "robust"):
        own = float(noise_floor(draw, far=0.05, null=getattr(null_providers, name), s=sv_null))
        assert rep[name]["K_null"] == int((sv_null > own).sum()), name


def test_the_sampled_null_plugs_into_the_library_contract():
    """`sampled_provider` is the module's public adapter onto `floor_from_null_sampler`, so it can
    front any entroptics read that takes a null. It was exported and never called by a test."""
    from entroptics.projection import noise_floor

    rng = np.random.default_rng(4)
    d, k = 128, 6
    u = np.linalg.qr(rng.standard_normal((d, k)))[0]
    v = np.linalg.qr(rng.standard_normal((d, k)))[0]
    A = (u * 8.0) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d)

    floor = float(noise_floor(A, far=0.05, null=je.sampled_provider("sign_flip", draws=40)))
    assert floor > 0.0
    # It must find the planted structure and stop there, like the direct sampled floor does.
    assert 1 <= int((np.linalg.svd(A, compute_uv=False) > floor).sum()) <= k

    with pytest.raises(ValueError, match="unknown surrogate"):
        je.sampled_provider("not_a_surrogate")
