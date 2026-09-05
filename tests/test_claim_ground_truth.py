"""The corrected read is invariant to the identity. The raw read is not.

This is the answer to the only serious objection the claim faces: you removed something and got a
bigger number, so why is bigger right? Because on a matrix whose rank is known by construction,
adding identity does not move the corrected read and destroys the raw one.

Nothing here needs a lens or a model. The ground truth is planted, so this runs everywhere and is
the most direct evidence in the suite that the claim's direction is correct rather than merely
different.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_jlens as je

D = 256


def planted(rank: int, alpha: float, seed: int) -> np.ndarray:
    """A matrix of known rank plus noise plus a known identity."""
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.standard_normal((D, rank)))[0]
    v = np.linalg.qr(rng.standard_normal((D, rank)))[0]
    signal = (u * np.linspace(12.0, 6.0, rank)) @ v.T
    return signal + rng.standard_normal((D, D)) / np.sqrt(D) + alpha * np.eye(D)


def ranks(A: np.ndarray) -> tuple[int, int]:
    dec = je.decompose(A)
    k_raw = je.transport_spectrum(A, null="mp", s=np.linalg.svd(A, compute_uv=False)).K
    k_dec = je.transport_spectrum(dec.residual, null="mp",
                                  s=np.linalg.svd(dec.residual, compute_uv=False)).K
    return k_raw, k_dec


@pytest.mark.parametrize("rank", [5, 10, 20])
def test_the_corrected_read_is_invariant_to_the_identity(rank):
    """The whole claim in one assertion. The planted rank does not change when alpha does, so a
    correct read must not either."""
    seen = {}
    for alpha in (0.0, 0.5, 1.5, 3.0, 6.0):
        _, k_dec = ranks(planted(rank, alpha, seed=rank * 100))
        seen[alpha] = k_dec
    assert set(seen.values()) == {rank}, (
        f"planted rank {rank} should be recovered at every identity strength; got {seen}")


@pytest.mark.parametrize("rank", [5, 10, 20])
def test_the_raw_read_collapses_as_the_identity_grows(rank):
    """The premise for the test above. If the raw read were also invariant there would be nothing
    to correct, and the claim would be empty."""
    weak, _ = ranks(planted(rank, 0.0, seed=rank * 100))
    strong, _ = ranks(planted(rank, 6.0, seed=rank * 100))
    assert weak == rank, f"with no identity the raw read should be right; got {weak}"
    assert strong < rank, f"with a large identity it should not be; got {strong}"
    assert strong < weak, f"raw {weak} -> {strong} as alpha rises"


def test_the_raw_read_only_ever_understates():
    """The direction of the error, which is what makes "understates" the right verb in the claim.
    A read that erred both ways would be noise rather than contamination."""
    for rank in (5, 10, 20):
        for alpha in (0.0, 0.5, 1.5, 3.0, 6.0):
            k_raw, _ = ranks(planted(rank, alpha, seed=rank * 100))
            assert k_raw <= rank, f"rank {rank}, alpha {alpha}: raw read OVERSTATED at {k_raw}"


def test_the_corrected_read_is_more_accurate_over_the_whole_sweep():
    """The summary figure the README quotes: 15 of 20 exact against 11 of 20, mean absolute error
    2.50 against 5.70. Recomputed rather than asserted from the table."""
    raw_err, dec_err = [], []
    for rank in (5, 10, 20, 40):
        for alpha in (0.0, 0.5, 1.5, 3.0, 6.0):
            k_raw, k_dec = ranks(planted(rank, alpha, seed=rank * 100 + int(alpha * 10)))
            raw_err.append(abs(k_raw - rank))
            dec_err.append(abs(k_dec - rank))
    raw_err, dec_err = np.array(raw_err), np.array(dec_err)

    assert (dec_err == 0).sum() > (raw_err == 0).sum(), (
        f"corrected exact {(dec_err == 0).sum()}, raw exact {(raw_err == 0).sum()}")
    assert dec_err.mean() < raw_err.mean() / 2, (
        f"corrected mean error {dec_err.mean():.2f}, raw {raw_err.mean():.2f}")


def test_the_corrected_reads_remaining_error_is_the_floors_own_and_not_the_identitys():
    """At planted rank 40 the corrected read saturates around 30 -- the `mp` floor's documented
    tendency to under-count signal-dense matrices, present with or without an identity. It is
    asserted here so the residual error is not read as a failure of the correction: it stays FLAT
    across alpha, which contamination would not.
    """
    got = [ranks(planted(40, alpha, seed=4000 + int(alpha * 10)))[1]
           for alpha in (0.0, 1.5, 6.0)]
    assert max(got) - min(got) <= 1, f"the corrected read should be flat across alpha; got {got}"
    assert all(g < 40 for g in got), "this fixture is meant to sit at the floor's own limit"
