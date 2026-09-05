"""Distribution-free floors for a transport, and the calibration check that picks one.

The library's default ``mp`` provider is the finite-size Johnstone / Tracy-Widom edge of an
i.i.d. **Gaussian** ensemble, and ``robust`` is documented as a Tukey fence -- "a heuristic
outlier fence, not a calibrated null". Both fall short of a floor for a real transport, and the
measurement that says so is in this module: a correctly calibrated provider, applied to a draw
from its own null, resolves ~0 modes. ``mp`` applied to a shuffled transport resolves 50-109.

Why the entry shuffle is the wrong null
---------------------------------------
A transport's heavy tails sit on a handful of coordinates, where i.i.d. heavy tails spread --
gpt2 layers 5 and 10 both attain their maximum at ``(447, 138)``; Qwen3.5-4B layers 0 and 12
both attain theirs at the **diagonal** entry ``(510, 510)``, and layer 30 at ``(795, 795)``. The
top 1% of rows carry 49% of the energy at gpt2 layer 10. These are the residual stream's
massive-activation dimensions, inherited by the transport.

Permuting entries over the whole matrix spreads those coordinates uniformly and manufactures a
bulk the transport does not have. The null has to preserve what is not in question -- the
magnitude profile, hence every row and column energy -- and destroy only what is: the sign and
position coherence that turns a magnitude profile into rank.

Three surrogates, by what they hold fixed
-----------------------------------------
  ``sign_flip``          |J| entrywise, EXACTLY. Every row energy, every column energy, the
                         Frobenius norm and the whole magnitude distribution survive; only the
                         sign coherence dies. The tightest entropy-matched null available for a
                         dense operator, and the default here.
  ``within_row_shuffle`` each row's multiset, hence every row energy. Column alignment dies.
  ``within_col_shuffle`` each column's multiset, hence every column energy. Row alignment dies.

Each becomes a provider through the library's own contract (``floor_from_null_sampler``): the
caller owns the null mechanism, the library owns the quantile. This module supplies a fast
top-singular-value scorer so ``draws >> 1/far`` is affordable at ``d_model`` scale; it is checked
against the library's exact scorer in ``tests/test_nulls.py``.

Where the surrogates apply
--------------------------
**All three collapse on a matrix whose energy is carried by a handful of cells.** An entry
shuffle spreads the massive-activation coordinates and manufactures a bulk, so its floor sits
low. Preserving the magnitude profile EXACTLY means
the giant cell appears in every surrogate draw, and one giant cell carries a large top singular
value on its own -- so the floor rises above the transport's real structure. The cure overshoots
the disease by more than the disease.

On a clean planted rank-6 at ``d = 256`` every surrogate is right, and ``sign_flip`` is both valid
and calibrated (exceedance 0.045 against a nominal 0.05).  Setting **one** cell of that same
matrix to 400, against an rms of 1.12:

    surrogate            K clean    K with one outlier cell
    sign_flip                  6                          1
    within_col_shuffle         6                          1
    within_row_shuffle         6                          0
    mp (analytic)              6                          7

    planted rank = 6

And a real transport is that second column. Removing the identity does not remove the massive
activations -- on gpt2's ``J - alpha*I`` the largest cell is 70x the rms at layer 0 rising to 599x
at layer 10, at ``(447, 138)`` throughout -- so ``sign_flip`` reads K = 2, 1, 1, 1 across layers
0/5/9/10 where ``mp`` reads 67, 51, 46, 39.

**So both nulls fail on massive activations, in opposite directions, and neither is right.**
``mp`` over-counts because a shuffle-flattened bulk sets its variance too low, the sampled
surrogates under-count because one preserved cell sets their floor too high. Nothing in this
module resolves that, and inventing a winsorised surrogate to close it would be fitting.

**Practical guidance.** Use these surrogates on a matrix whose energy is not dominated by a
handful of cells, where they are calibrated and they work. On a real language-model transport:
``decompose`` first, read ``mp`` knowing it is a lower bound, and read ``PR`` beside ``K`` so a
concentrated spectrum is visible. Treat a sampled-floor ``K`` of 0-1 on a transport as a
statement about that transport's largest entry, not about its rank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from entroptics import null_providers

from .frames import as_frame


# ── surrogates: what each one holds fixed ────────────────────────────────────────────────

def sign_flip(J, rng: np.random.Generator) -> np.ndarray:
    """``J`` with i.i.d. random signs. ``|J|`` is preserved entrywise and exactly."""
    A = np.asarray(J)
    return A * rng.choice(np.array([-1.0, 1.0]), size=A.shape)


def within_row_shuffle(J, rng: np.random.Generator) -> np.ndarray:
    """Each row permuted independently: row energies preserved exactly."""
    A = np.asarray(J)
    idx = np.argsort(rng.random(A.shape), axis=1)
    return np.take_along_axis(A, idx, axis=1)


def within_col_shuffle(J, rng: np.random.Generator) -> np.ndarray:
    """Each column permuted independently: column energies preserved exactly."""
    A = np.asarray(J)
    idx = np.argsort(rng.random(A.shape), axis=0)
    return np.take_along_axis(A, idx, axis=0)


SURROGATES = {"sign_flip": sign_flip,
              "within_row": within_row_shuffle,
              "within_col": within_col_shuffle}


# ── the scorer: the top singular value, without a full SVD ───────────────────────────────

def top_singular(A: np.ndarray, *, block: int = 8, iters: int = 12,
                 rng: np.random.Generator | None = None) -> float:
    """Largest singular value by randomized subspace iteration.

    A sampled floor needs ``draws >> 1/far`` scores, and a full SVD at ``d = 2560`` is seconds
    apiece; this is milliseconds. Subspace iteration on ``AᵀA`` with a block converges on the
    top value even when the leading values are close, which plain power iteration does not.
    Checked against ``numpy.linalg.svd`` and against the library's own
    ``null_providers.top_spectrum_value`` in the tests.
    """
    A = np.asarray(A, dtype=np.float64)
    rng = np.random.default_rng(0) if rng is None else rng
    n = A.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((n, min(int(block), n))))
    for _ in range(int(iters)):
        Q, _ = np.linalg.qr(A.T @ (A @ Q))
    B = A @ Q                                     # (m, block): the compressed image
    return float(np.linalg.svd(B, compute_uv=False)[0])


# ── the floor, and the calibration that earns it ─────────────────────────────────────────

@dataclass(frozen=True)
class SampledFloor:
    """A distribution-free floor, the sample it came from, and its held-out calibration."""
    surrogate:  str
    far:        float
    draws:      int
    floor:      float
    tops:       np.ndarray        # the draws the floor's quantile was taken over
    check:      np.ndarray        # HELD-OUT draws, scored against the finished floor
    K:          int               # modes of J above this floor
    exceedance: float             # fraction of held-out draws above it -- should equal far

    def _tol(self) -> float:
        n = int(self.check.size)
        return 2.0 * float(np.sqrt(self.far * (1.0 - self.far) / max(n, 1))) + 1e-12

    @property
    def valid(self) -> bool:
        """Does the floor keep the promise that matters -- no MORE false alarms than claimed?

        This is the one-sided property, and it is the one a detection threshold has to have.
        Exceeding the nominal rate invalidates every count taken against the floor; falling
        below it costs sensitivity and nothing else.
        """
        return bool(self.check.size) and self.exceedance <= self.far + self._tol()

    @property
    def calibrated(self) -> bool:
        """The two-sided property: is the actual rate the claimed rate, within binomial error?

        Checked on draws the floor was NOT fitted to -- scoring against its own quantile sample
        returns ``far`` by construction and tests nothing. Expect this to be False more often
        than ``valid``: the empirical ``(1-far)`` quantile of a right-skewed null is a
        high-variance estimator at only a few hundred draws, and it lands high more often than
        low. Measured on a planted rank-6 transport at 200 draws, exceedance came out 0.005
        against a nominal 0.05 -- conservative by an order of magnitude, and still resolving all
        6 planted modes. Raise ``draws`` to tighten it; the library's own guidance is that
        resolving a level ``far`` needs ``draws >> 1/far``.
        """
        return bool(self.check.size) and abs(self.exceedance - self.far) <= self._tol()


def sampled_floor(J, *, surrogate: str = "sign_flip", far: float = 0.05, draws: int = 200,
                  check_draws: int | None = None, seed: int = 0,
                  s: np.ndarray | None = None) -> SampledFloor:
    """The ``(1 - far)`` quantile of the top singular value over ``draws`` surrogate draws.

    No distributional assumption and no fitted constant: the null is ``J``'s own magnitude
    profile with its coherence destroyed, and the threshold is a quantile of that null. This is
    the library's sampled-null contract (``floor_from_null_sampler``) with a fast scorer.
    """
    if surrogate not in SURROGATES:
        raise ValueError(f"unknown surrogate {surrogate!r}; available: {sorted(SURROGATES)}")
    A = as_frame(J, name="J")
    fn = SURROGATES[surrogate]
    rng = np.random.default_rng(seed)
    n_check = int(draws) if check_draws is None else int(check_draws)
    tops = np.array([top_singular(fn(A, rng), rng=rng) for _ in range(int(draws))])
    floor = float(np.quantile(tops, 1.0 - float(far)))
    sv = np.linalg.svd(A, compute_uv=False) if s is None else np.asarray(s, dtype=np.float64)
    # Held out: these draws did not shape the floor, so their exceedance rate is a real test.
    check = np.array([top_singular(fn(A, rng), rng=rng) for _ in range(n_check)])
    return SampledFloor(surrogate=surrogate, far=float(far), draws=int(draws), floor=floor,
                        tops=tops, check=check, K=int((sv > floor).sum()),
                        exceedance=float((check > floor).mean()) if n_check else float("nan"))


def provider(surrogate: str = "sign_flip", *, draws: int = 200, far: float | None = None):
    """The same null as a library provider, for ``noise_floor(..., null=provider(...))``.

    Routes through ``null_providers.floor_from_null_sampler``, so it plugs into any entroptics
    read that takes a null -- ``Projection``, ``Screen``, the certificate. Uses the library's
    exact scorer rather than the fast one, because a provider is called once per read and
    exactness costs nothing there.
    """
    if surrogate not in SURROGATES:
        raise ValueError(f"unknown surrogate {surrogate!r}; available: {sorted(SURROGATES)}")
    return null_providers.floor_from_null_sampler(SURROGATES[surrogate], draws=draws, far=far)


def calibration_report(J, *, far: float = 0.05, draws: int = 200, seed: int = 0) -> dict:
    """Every candidate floor on one transport, with the check that decides between them.

    For each: the floor, ``K`` on ``J``, and ``K_null`` -- the whole procedure re-run on a draw
    from the null, floor included. A provider with ``K_null > 0`` is mis-calibrated on this object
    and its ``K`` is not a count of structure. This is a measurement, and every quantity in it is
    read rather than fitted.

    ``K_null`` counts against ``floor_null``, the floor computed ON the null draw, and not against
    the floor computed on ``J``. Calibration is "apply the provider to noise and it finds nothing",
    so both halves have to come from the noise. For ``mp`` the two floors coincide exactly --
    ``sign_flip`` preserves ``|J|`` entrywise, so the per-cell variance and hence the MP edge are
    identical. For ``robust`` they do not: the Tukey fence is a quantile of the SPECTRUM, which
    the sign flip
    changes, and at gpt2 layer 0 the old reading gave ``K_null`` 117 against the correct 11.
    """
    from entroptics.projection import noise_floor

    A = as_frame(J, name="J")
    sv = np.linalg.svd(A, compute_uv=False)
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("mp", "robust"):
        f = float(noise_floor(A, far=far, null=getattr(null_providers, name), s=sv))
        null_draw = SURROGATES["sign_flip"](A, rng)
        sv_null = np.linalg.svd(null_draw, compute_uv=False)
        f_null = float(noise_floor(null_draw, far=far,
                                   null=getattr(null_providers, name), s=sv_null))
        out[name] = {"floor": f, "K": int((sv > f).sum()),
                     "K_null": int((sv_null > f_null).sum()),
                     "floor_null": f_null, "sampled": False}
    for name in SURROGATES:
        sf = sampled_floor(A, surrogate=name, far=far, draws=draws, seed=seed, s=sv)
        out[name] = {"floor": sf.floor, "K": sf.K, "exceedance": sf.exceedance,
                     "calibrated": sf.calibrated, "sampled": True}
    return out
