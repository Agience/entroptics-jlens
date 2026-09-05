"""B2 again, against the coverage metric rather than the complement read.

The complement read measured mode counts after projection, and the re-read whitens per channel,
which is not invariant under projection. Coverage measures the subspace overlap directly and
compares it with the chance level for a readout of the same size.
"""
import numpy as np
import pytest

import entroptics_jlens as je


def stream(d=256, T=128, k_true=24, seed=0):
    rng = np.random.default_rng(seed)
    V = je.haar_orthogonal(d, rng)[:, :k_true]
    return (rng.standard_normal((T, k_true)) @ (V * np.linspace(12.0, 4.0, k_true)).T
            + rng.standard_normal((T, d)))


@pytest.mark.parametrize("k", [2, 5, 10])
def test_coverage_of_the_signals_own_top_k_is_k_over_k_signal(k):
    """A readout that is exactly the projector onto the signal's own top-k directions covers
    exactly those k of the signal's resolved directions -- so coverage is k / k_signal."""
    H = stream()
    Vt = np.linalg.svd(je.as_frame(H), full_matrices=False)[2]
    P = Vt[:k].T @ Vt[:k]
    cov = je.coverage(H, H @ P.T)
    assert cov.k_readout == k
    assert cov.coverage == pytest.approx(k / cov.k_signal, abs=0.02), (
        f"k={k}: coverage {cov.coverage:.3f}, expected {k / cov.k_signal:.3f}")
    assert cov.above_chance


def test_full_span_is_coverage_one():
    """A readout spanning the whole signal subspace covers all of it."""
    H = stream()
    cov = je.coverage(H, H)
    assert cov.coverage == pytest.approx(1.0, abs=1e-6)
    assert cov.excess == pytest.approx(1.0, abs=1e-6)


def test_a_random_readout_lands_at_chance_not_at_zero():
    """The point the complement read missed: a random readout does not cover *nothing*, it
    covers k_readout/d. Reporting the residue as a finding is what went wrong before."""
    H = stream()
    rng = np.random.default_rng(1)
    d = H.shape[1]
    k = 12
    Q = je.haar_orthogonal(d, rng)[:, :k]
    cov = je.coverage(H, H @ (Q @ Q.T).T)
    assert cov.coverage == pytest.approx(cov.null, abs=3 * cov.null), (
        f"random readout covered {cov.coverage:.4f} against a null of {cov.null:.4f}")
    assert not cov.above_chance
    assert abs(cov.excess) < 0.15


def test_partial_overlap_is_reported_proportionally():
    """Half the readout's directions inside the signal's resolved set, half outside: coverage
    should count the inside half only."""
    H = stream()
    Vt = np.linalg.svd(je.as_frame(H), full_matrices=False)[2]
    k = 8
    B = np.concatenate([Vt[: k // 2].T, Vt[-(k // 2):].T], axis=1)
    Q, _ = np.linalg.qr(B)
    cov = je.coverage(H, H @ (Q @ Q.T).T)
    ks = cov.k_signal
    assert cov.coverage == pytest.approx((k // 2) / ks, abs=0.03), (
        f"coverage {cov.coverage:.3f}, expected {(k // 2) / ks:.3f}")


def test_the_sampled_null_brackets_the_analytic_one():
    """The analytic null is a mean; a decision needs its spread."""
    H = stream()
    k = 12
    draws = je.coverage_null_sample(H, k, draws=48, seed=0)
    analytic = k / H.shape[1]
    assert draws.mean() == pytest.approx(analytic, rel=0.35)
    assert draws.std() > 0


def test_coverage_falls_when_the_transport_loses_rank():
    """B4' in miniature: a transport truncated below the structure it carries covers less.

    The calibrated version measures how small a gap this survives (see
    ``research/experiments/exp13_ranking_sensitivity.py``); this pins the direction, which is the part a
    regression can break silently. Coverage is the only read of the three tested that points the
    same way for both degradation modes, so the sign is the load-bearing property.
    """
    H = stream()
    d = H.shape[1]
    rng = np.random.default_rng(3)
    V = je.haar_orthogonal(d, rng)[:, :32]
    J = (V * np.linspace(6.0, 1.0, 32)) @ V.T
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    J_cut = (U[:, :8] * s[:8]) @ Vt[:8]
    assert je.coverage(H, H @ J.T).excess > je.coverage(H, H @ J_cut.T).excess


def test_coverage_falls_when_the_transport_gains_noise():
    """The opposite failure mode: directions added that carry nothing."""
    H = stream()
    d = H.shape[1]
    rng = np.random.default_rng(4)
    V = je.haar_orthogonal(d, rng)[:, :32]
    J = (V * np.linspace(6.0, 1.0, 32)) @ V.T
    G = rng.standard_normal(J.shape)
    J_noisy = J + 0.6 * (np.linalg.norm(J) / np.linalg.norm(G)) * G
    assert je.coverage(H, H @ J.T).excess > je.coverage(H, H @ J_noisy.T).excess


def test_coverage_exposes_no_reliability_predicate():
    """Two were tried and both withdrawn; the second withdrawal is the finding.

    Reliability is not recoverable from the coverage values. Separations below 0.001 in `excess`
    order correctly 89% of the time across the calibrated degradations, while the catalogue's two
    published fits sit at the same separations and order backwards every time. No threshold on the
    separation can tell those apart -- what governs is the kind of difference, which the caller
    knows and this object does not. A predicate here would be a guarantee the read cannot keep.
    """
    H = stream()
    c = je.coverage(H, H)
    assert not hasattr(c, "resolves_gap")
    assert not hasattr(c, "reliable_for_ranking")
