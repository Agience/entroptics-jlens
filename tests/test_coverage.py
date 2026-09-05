"""B2 -- analytic coverage. Does the complement read report a KNOWN overlap correctly?

It does not. These are marked ``xfail(strict=True)``: the benchmark stays live, its failure is
recorded rather than deleted, and if the read is ever fixed the strict marker turns the
pass into a reported XPASS instead of silently going green.

B1 (planted rank) and B3 (matched-spectrum null) both test that the instrument does not fire on
noise. B2 tests the harder direction: that it fires *correctly* when there is an exact, known
amount to find.

Construction. Take a stream ``H`` and build a transport that is exactly the orthogonal projector
onto ``H``'s own top-``k`` right singular directions. That transport provably carries those k
directions and nothing else, so the complement -- what the stream resolves once the transport's
directions are removed -- must fall by exactly k.

This is the benchmark that B3's failure made decisive. `outside% ~ 100%` under a random rank-K
map is what a read returns when it cannot see overlap at all; a read that also returns ~100%
here, where the overlap is total by construction, can only ever say "nothing overlaps".
"""
import numpy as np
import pytest
from entroptics import Projection

import entroptics_jlens as je


def stream(d=256, T=128, k_true=24, seed=0):
    """A stream with a controlled number of strong directions above the floor."""
    rng = np.random.default_rng(seed)
    V = je.haar_orthogonal(d, rng)[:, :k_true]
    amp = np.linspace(12.0, 4.0, k_true)
    return rng.standard_normal((T, k_true)) @ (V * amp).T + rng.standard_normal((T, d))


def projector_onto_top_k(H, k):
    """The exact orthogonal projector onto H's top-k right singular directions."""
    Vt = np.linalg.svd(je.as_frame(H), full_matrices=False)[2]
    Vk = Vt[:k].T
    return Vk @ Vk.T


def test_transport_that_is_a_projector_carries_exactly_k_directions():
    """Sanity on the construction itself before it is used to judge the read."""
    H = stream()
    for k in (1, 5, 20):
        P = projector_onto_top_k(H, k)
        assert np.allclose(P, P.T, atol=1e-10)                 # symmetric
        assert np.allclose(P @ P, P, atol=1e-9)                # idempotent
        assert round(float(np.trace(P))) == k                  # rank k
        # and it reproduces the stream's own top-k component
        assert np.allclose(H @ P.T, H @ P, atol=1e-9)


@pytest.mark.xfail(strict=True, reason="B2: the complement read cannot report a known overlap. It also fails its own matched-spectrum null (PAPER.md 7.0). Kept as a live benchmark: strict=True means fixing the read turns this into an XPASS and says so.")
@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_complement_falls_by_exactly_k(k):
    """The decisive assertion: removing a subspace the stream provably resolves must reduce the
    complement's resolved count by exactly the size of that subspace."""
    H = stream()
    K_stream = Projection(H).K_signal
    assert K_stream > k, f"benchmark needs K_stream ({K_stream}) > k ({k})"
    ls = je.layer_screen(H, projector_onto_top_k(H, k), layer=0, rank=None)
    K_comp = je.complement(ls).K_signal
    assert K_comp == K_stream - k, (
        f"k={k}: stream resolves {K_stream}, complement resolves {K_comp}, "
        f"expected {K_stream - k}. A read that cannot report a known overlap cannot report "
        f"an unknown one.")


@pytest.mark.xfail(strict=True, reason="B2: the complement read cannot report a known overlap. It also fails its own matched-spectrum null (PAPER.md 7.0). Kept as a live benchmark: strict=True means fixing the read turns this into an XPASS and says so.")
def test_the_random_control_does_not_move_the_complement():
    """The B3 comparison, in the same construction: a random rank-k map of the same rank leaves
    the complement where it was, while the projector removes exactly k. If both behaved the same
    the read would carry no information."""
    H = stream()
    rng = np.random.default_rng(1)
    K_stream = Projection(H).K_signal
    k = 8
    real = je.layer_screen(H, projector_onto_top_k(H, k), layer=0, rank=None)
    Q = je.haar_orthogonal(H.shape[1], rng)[:, :k]
    rand = je.layer_screen(H, Q @ Q.T, layer=0, rank=None)
    K_real, K_rand = je.complement(real).K_signal, je.complement(rand).K_signal
    assert K_real == K_stream - k
    assert K_rand > K_real, (
        f"a random rank-{k} projector removed as much as the aligned one "
        f"({K_rand} vs {K_real} of {K_stream}); the read cannot distinguish alignment")


@pytest.mark.xfail(strict=True, reason="B2: the complement read cannot report a known overlap. It also fails its own matched-spectrum null (PAPER.md 7.0). Kept as a live benchmark: strict=True means fixing the read turns this into an XPASS and says so.")
def test_partial_overlap_is_reported_proportionally():
    """Half the transport's directions inside the stream's resolved set, half outside: the
    complement should fall by the inside half only."""
    d, k = 256, 8
    H = stream(d=d)
    K_stream = Projection(H).K_signal
    Vt = np.linalg.svd(je.as_frame(H), full_matrices=False)[2]
    inside = Vt[: k // 2].T                       # top directions: resolved
    outside = Vt[-(k // 2):].T                    # tail directions: not resolved
    B = np.concatenate([inside, outside], axis=1)
    Q, _ = np.linalg.qr(B)
    ls = je.layer_screen(H, Q @ Q.T, layer=0, rank=None)
    K_comp = je.complement(ls).K_signal
    assert K_comp == K_stream - k // 2, (
        f"expected the complement to fall by the {k // 2} resolved directions only, "
        f"got {K_stream} -> {K_comp}")
