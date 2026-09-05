"""The spectral read of one Jacobian-lens transport ``J_l``.

``J_l = E[d h_final / d h_l]`` is a dense ``(d_model, d_model)`` average.  The Jacobian lens
uses it whole; the number the paper reports for workspace capacity ("on the order of tens",
~25 by sparse decomposition) comes from a chosen activity threshold applied downstream, not
from ``J`` itself.

Entroptics reads the rank ``J`` actually resolves straight off its singular spectrum, against a
derived edge, with no fitted constant.

**Read it on ``J - alpha*I``, not on ``J``.**  ``decompose`` comes first, and this is not a
stylistic preference.  The floor is estimated from the spectrum it is judging, so the identity
raises the floor by the same amount it flattens the spectrum, and the modes underneath disappear.
Measured against the identity share at each layer:

    Qwen3.5-4B  layer  0    K(J) =  94    K(J - alpha*I) =  93    identity 0.008
                layer 12    K(J) = 203    K(J - alpha*I) = 178    identity 0.070
                layer 24    K(J) =  49    K(J - alpha*I) = 131    identity 0.583
                layer 30    K(J) =  25    K(J - alpha*I) = 183    identity 0.790

``mp`` null, ``far=0.05``, same matrix on both sides.

**The effect changes sign, so it is not a correction that can be applied afterwards.**  Where the
identity is small, removing it takes real energy out and ``K`` falls a little (203 -> 178 at layer
12).  Where the identity dominates, removing it uncovers modes that were beneath the floor it was
holding up, and ``K`` rises by 7.3x (25 -> 183 at layer 30).  The crossover is somewhere in
between and nothing here locates it.  What follows is only that the two are different
measurements and the decomposed one is the one about the transport -- not that a single factor
relates them.

On an identity-cored fixture with rank 8 planted, ``J`` resolves 0 and ``J - alpha*I`` resolves
exactly 8.  ``entroptics-jlens audit`` prints both columns for this reason.

The read is a spectral statement about a dense matrix and needs no ordered axis -- unlike
``Projection``, which whitens per channel and folds along an ordered axis, and is the right
front door for a residual stream but not for ``J``.

Which null, and why the choice carries the result
--------------------------------------
``K = #(s_k > floor)`` and the floor comes from a null provider.  The default ``mp`` provider
estimates its per-cell variance from the whole matrix -- signal included -- so a matrix carrying
a lot of structure lifts its own floor and hides its weakest modes.  Measured here on planted
ranks at ``d = 256``, every mode supra-threshold by the BBP criterion:

    planted   amplitude span   K under mp   K under robust
       30       [40, 400]          19             29
       30       [40, 120]          22              -
       30       [40,  60]          25              -
       10       [40, 120]          10             10

So ``mp`` under-counts, and the loss grows with both the number of modes and their spread; a
rank-10 case is unaffected.  **``K`` is a lower bound on the resolved rank**, and the bound is
loosest exactly where the answer matters -- a real transport reading ``K ~ 25`` could be
carrying more.

**``robust`` is not a second opinion about the same quantity, and the sentence that used to
stand here said it was.**  It said ``robust`` "recovers weak modes the signal's own energy would
otherwise hide", which is what the planted-rank table above shows at ``d = 256`` and is false on
real transports.  ``null_providers.robust`` is the Tukey upper fence ``Q3 + 1.5*(Q3 - Q1)`` of
the SINGULAR SPECTRUM, and its own docstring in the library calls it "a heuristic outlier fence
for heavy-tailed spectra (not a calibrated null)".  Two consequences, on gpt2's identity-free
transports:

  * **It carries no false-alarm rate, and ``far`` does not reach it.**  Sweeping ``far`` from 0.5
    to 1e-6 at layer 5 moves the ``mp`` floor 2.136 -> 2.196 and ``K`` 53 -> 47, and leaves the
    robust floor at 2.40679 -- identical to five decimals at every value.  ``TransportSpectrum.far``
    still records what the caller passed, so a robust reading travels with a number that did not
    apply to it.

  * **It reads FEWER modes than ``mp`` wherever the spectrum is heavy-tailed**, which is most of a
    real lens.  Layer 0 gives mp 67 / robust 104, and from layer 2 up the order reverses and the
    gap widens with excess kurtosis: 59/51 at layer 2 (kurtosis 23), 48/22 at layer 7 (10,709),
    46/13 at layer 9 (68,163).  A few dominant directions raise Q3 and the IQR together, so the
    fence rises with the very structure it is supposed to be separating from noise.

Report both, and read the gap as a statement about the SHAPE of the spectrum rather than as a
confidence interval on a count.  Only ``mp`` answers "how many modes would noise have produced".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from entroptics import null_providers
from entroptics.projection import mode_significance, noise_floor

from .frames import as_frame

NULLS = {"mp": null_providers.mp, "robust": null_providers.robust}


@dataclass(frozen=True)
class TransportSpectrum:
    """What ``J`` resolves, and the evidence behind it.

    ``K`` is the resolved rank under ``null_name`` at the reader's ``far``, by the library's own
    definition ``K = #(s_k > floor)`` -- with the caveat that ``far`` reaches the ``mp`` provider
    and NOT ``robust``, whose fence is a quantile of the spectrum and takes no false-alarm rate.
    ``far`` records what the caller passed either way, so read it together with ``null_name``.

    ``energy_resolved`` is the fraction of ``||J||_F^2``
    those ``K`` modes carry.  ``deviate`` / ``pvalue`` are the per-mode Tracy-Widom evidence
    under the DEFAULT ``mp`` null -- ``mode_significance`` takes no provider -- so ``recount``
    re-counts against ``mp`` regardless of which provider fixed ``K``.
    """
    shape:           tuple[int, int]
    far:             float
    null_name:       str
    excess_kurtosis: float
    singular:        np.ndarray
    deviate:         np.ndarray
    pvalue:          np.ndarray
    floor:           float
    K:               int
    energy_resolved: float

    @property
    def saturated(self) -> bool:
        """Every mode resolved.  The truncation is then the identity, ``J`` has no null space,
        and every complement read built on it is vacuous -- a result to report rather than route
        around.  Not reachable by any matrix constructed so far: concentrating a spectrum lifts
        the self-estimated floor with it, and a perfectly flat spectrum (a scaled rotation) sits
        exactly at the null with every p-value 1.0."""
        return self.K >= min(self.shape)

    def recount(self, far: float) -> int:
        """The resolved rank at another false-alarm level, from the ``mp`` evidence."""
        return int((self.pvalue < float(far)).sum())


#: The fourth standardised moment of a Gaussian. Subtracting it gives EXCESS kurtosis, so
#: zero means "as heavy-tailed as a Gaussian". Fixed by the distribution, not chosen.
GAUSSIAN_KURTOSIS = 3.0



def _excess_kurtosis(A: np.ndarray) -> float:
    """Excess kurtosis of the cells, from raw moments rather than centred powers.

    The obvious spelling, ``((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3``, allocates three
    temporaries the size of the matrix and was **90% of this function's runtime**: 118 ms of 132 ms
    at ``d = 768``, for a number that plays no part in ``K``. Expanding the central moment into raw
    ones costs two dot products and one temporary:

        mu4 = m4 - 4*m1*m3 + 6*m1^2*m2 - 3*m1^4

    Measured 15x faster and identical to 4.4e-13 relative on a 768x768 draw. The dot products also
    keep the accumulation in one BLAS pass rather than three numpy ones.
    """
    x = A.reshape(-1)
    n = x.size
    m1 = float(x.mean())
    m2 = float(x.dot(x)) / n
    var = m2 - m1 * m1
    if var <= 0.0:
        return 0.0
    x2 = x * x
    m3 = float(x.dot(x2)) / n
    m4 = float(x2.dot(x2)) / n
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
    return float(mu4 / var ** 2 - GAUSSIAN_KURTOSIS)


def transport_spectrum(J, *, far: float = 0.05, null: str = "mp", seed: int = 0,
                       s: np.ndarray | None = None) -> TransportSpectrum:
    """Read ``J``'s singular spectrum against a derived noise edge.

    ``null`` names a provider in ``NULLS``. Unknown names refuse rather than falling back to the
    default: a run that silently swapped the null would report two incomparable numbers under
    one column heading.

    ``s`` is the precomputed singular spectrum. The SVD is the whole cost of this read at
    ``d_model`` scale, and it does not depend on the provider, so a caller reading one matrix
    under several nulls computes it once (see ``spectrum_under_nulls``).
    """
    if null not in NULLS:
        raise ValueError(f"unknown null {null!r}; available: {sorted(NULLS)}")
    A = as_frame(J, name="J")
    s = np.linalg.svd(A, compute_uv=False) if s is None else np.asarray(s, dtype=np.float64)
    ms = mode_significance(A, s)
    kurt = _excess_kurtosis(A)
    floor = float(noise_floor(A, far=far, null=NULLS[null], s=s, seed=seed))
    K = int((s > floor).sum())
    total = float((s ** 2).sum())
    return TransportSpectrum(
        shape=(int(A.shape[0]), int(A.shape[1])), far=float(far), null_name=null,
        excess_kurtosis=kurt, singular=s, deviate=ms.deviate, pvalue=ms.pvalue, floor=floor, K=K,
        energy_resolved=(float((s[:K] ** 2).sum()) / total) if total > 0.0 else 0.0,
    )


def spectrum_under_nulls(J, *, far: float = 0.05, seed: int = 0) -> dict:
    """Every available provider, on one matrix.

    ``mp`` is the calibrated null and a lower bound on the resolved rank.  ``robust`` is a Tukey
    fence on the spectrum, not a null: it ignores ``far`` entirely and reads fewer modes than
    ``mp`` wherever the spectrum is heavy-tailed.  The spread between them describes the shape of
    the spectrum; it is not an interval on a count.  See this module's header for the
    measurement."""
    A = as_frame(J, name="J")
    sv = np.linalg.svd(A, compute_uv=False)          # provider-independent: computed once
    return {name: transport_spectrum(A, far=far, null=name, seed=seed, s=sv) for name in NULLS}


def resolved_transport(J, *, far: float = 0.05, null: str = "mp", seed: int = 0):
    """``J`` truncated at the rank it resolves, with the spectrum that fixed the rank.

    Returns ``(J_K, spectrum)``.  ``J_K`` has exact rank ``spectrum.K``, so ``pinv(J_K) @ J_K``
    is the orthogonal projector onto the resolved row space and the transport's null space is
    real.  A transport that resolves nothing is a collapse and raises: there is no rank to
    truncate at, and returning ``J`` whole would report a full-rank map as a resolved one.

    The default ``mp`` null truncates conservatively -- a lower-bound rank means a larger null
    space, so ``certify`` and the complement read attribute to the transport everything it might
    still carry.  Pass ``null="robust"`` for the tighter truncation and report both.
    """
    A = as_frame(J, name="J")
    spec = transport_spectrum(A, far=far, null=null, seed=seed)
    if spec.K == 0:
        raise ValueError(
            f"transport resolves 0 modes above the noise floor at far={far}, null={null!r} "
            f"(top singular value {spec.singular[0]:.6g} vs floor {spec.floor:.6g}): there is no "
            f"resolved rank to truncate at. This is a reading about J, not a recoverable error.")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    K = spec.K
    return (U[:, :K] * s[:K]) @ Vt[:K], spec

#: How far above the arithmetic limit the floor must sit before the cheap spectrum is trusted for
#: it. `energy_spectrum`'s eigvalsh route halves the significant digits of the SMALLEST singular
#: values, which costs roughly 1e8 there; a margin of 1e10 leaves two orders of headroom on top of
#: that. Measured on ten published lenses, the smallest actual margin is 1.9e12, so real transports
#: clear this by two further orders and the exact route is never taken for them. It exists for the
#: matrix that does not.
FLOOR_PRECISION_MARGIN = 1e10


def spectrum_for_floor(J, *, far: float = 0.05, null: str = "mp", seed: int = 0):
    """``transport_spectrum`` via the cheap spectrum where that is safe, exactly where it is not.

    ``energy_spectrum``'s default route is ``eigvalsh(A^T A)``, about 3x faster than the SVD at
    ``d = 2560`` and documented as unsafe for a noise floor because squaring loses precision in the
    smallest singular values. That warning is about the region NEAR the arithmetic limit. The floor
    is not there: measured on ten published lenses it sits 1.9e12 to 2.4e13 times above
    ``eps * s[0] * sqrt(max(shape))``, so squaring cannot move it across a singular value, and
    ``K`` agrees with the exact route on 20 of 20 reads.

    So this takes the cheap route, checks the margin it actually got, and recomputes exactly if the
    margin is thin. The check costs one comparison; the fallback costs the SVD that would otherwise
    always have been paid.

    Returns ``(spectrum, used_exact)`` so a caller can report which route produced its numbers
    rather than leaving that to be assumed.
    """
    A = as_frame(J, name="J")
    from .spectra import energy_spectrum

    s_fast = energy_spectrum(A)
    if floor_is_resolvable(A, s_fast, far=far, null=null, seed=seed):
        return transport_spectrum(A, far=far, null=null, seed=seed, s=s_fast), False
    s_exact = np.linalg.svd(A, compute_uv=False)
    return transport_spectrum(A, far=far, null=null, seed=seed, s=s_exact), True


def floor_is_resolvable(J, s, *, far: float = 0.05, null: str = "mp", seed: int = 0) -> bool:
    """Does ``s`` place the noise floor far enough above the arithmetic limit to be trusted?

    The cheap spectrum route squares the singular values, which halves the significant digits of
    the smallest ones -- and the smallest ones are where a detection threshold lives. This asks
    whether that loss can reach the floor: it cannot if the floor sits ``FLOOR_PRECISION_MARGIN``
    above ``eps * s[0] * sqrt(max(shape))``.

    **The one place this question is answered.** ``spectrum_for_floor`` uses it for a single
    matrix, and the CLI uses it for the pair it reads from a shared Gram; a second copy of the
    comparison in the caller would be a second threshold to keep in step.
    """
    A = as_frame(J, name="J")
    sv = np.asarray(s, dtype=np.float64)
    limit = float(np.finfo(np.float64).eps * sv[0] * np.sqrt(max(A.shape)))
    if limit <= 0.0:
        return True
    floor = float(noise_floor(A, far=far, null=NULLS[null], s=sv, seed=seed))
    return floor / limit >= FLOOR_PRECISION_MARGIN
