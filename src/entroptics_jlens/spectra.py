"""Threshold-free reads of a spectrum, and a fast exact route to one.

These are the reads that survive on a corpus-averaged transport. A detection floor needs a noise
bulk whose edge it can find, and two independently published fits of Qwen3.5-4B correlate 0.9999
by layer 30 -- the averaging removed the noise, leaving a spectrum with no bulk. Participation ratio and
Shannon effective rank carry no false-alarm level and need no null, which is what makes them
appropriate here and what stops them answering "how many directions are real". Principal angles
between independent fits answer that instead.

Both are energy-weighted, so the large singular values determine them and the smallest
contribute negligibly -- which is why ``energy_spectrum`` is safe for them and unsafe
elsewhere.
"""
from __future__ import annotations

import numpy as np

from .frames import as_frame


def energy_spectrum(A, *, exact: bool = False) -> np.ndarray:
    """Descending singular values of ``A``.

    By default via ``eigvalsh(AᵀA)``, which is 3.2x faster than the SVD at ``d = 2560`` and
    agrees on ``participation_ratio`` and ``shannon_rank`` to a relative 1e-15 -- both weight by
    ``s²``, so the precision the squaring costs sits entirely in values those reads do not see.

    **Do not use this for a noise floor.** Squaring loses roughly half the significant digits in
    the smallest singular values, and that is exactly the region a detection threshold reads:
    ``mode_significance`` and ``noise_floor`` take the matrix and compute their own spectrum.
    ``exact=True`` forces the SVD when a caller wants one number from both routes.
    """
    M = as_frame(A, name="A")
    if exact:
        return np.linalg.svd(M, compute_uv=False)
    ev = np.linalg.eigvalsh(M.T @ M if M.shape[0] >= M.shape[1] else M @ M.T)
    return np.sqrt(np.clip(ev, 0.0, None))[::-1]


def participation_ratio(s: np.ndarray) -> float:
    """``(Σ s²)² / Σ s⁴`` -- an energy-weighted effective rank, no threshold in it.

    Equals ``r`` exactly for a spectrum with ``r`` equal non-zero values and ``1`` for a rank-1
    spectrum, so it reads as a dimension count without ever deciding what counts as noise."""
    s2 = np.asarray(s, dtype=np.float64) ** 2
    q = float((s2 ** 2).sum())
    return float((s2.sum()) ** 2 / q) if q > 0 else 0.0


def shannon_rank(s: np.ndarray) -> float:
    """``2^H(p)`` for ``p_k = s_k²/Σs²`` -- the effective rank the energy distribution's own
    Shannon entropy implies. Reads the whole spectrum's shape where the participation ratio
    reads its second moment, so the two disagree when a spectrum has a long weak tail."""
    s2 = np.asarray(s, dtype=np.float64) ** 2
    tot = float(s2.sum())
    if tot <= 0:
        return 0.0
    p = s2 / tot
    p = p[p > 0]
    return float(2.0 ** (-(p * np.log2(p)).sum()))


def principal_angles(A, B, k: int) -> np.ndarray:
    """Cosines of the principal angles between the top-``k`` right singular subspaces of ``A``
    and ``B``, descending.

    The reproducibility read: with two independent fits of the same model this says how far into
    the spectrum they still agree on a subspace, which is what a scalar noise floor cannot say.
    Follows the stability construction of Scanu et al. (arXiv:2606.09964, eq. 6) with *fit* in
    place of noise level.

    **A top-``k`` subspace exists only where the spectrum says which ``k`` directions are on top.**
    Two cases leave it undefined, and both refuse rather than return a number:

    * ``k`` past the matrix's numerical rank -- fewer than ``k`` singular values stand above
      round-off, so the remaining "top" directions are a basis of the null space, which is any
      basis at all;
    * ``k`` inside a run of equal singular values -- ``s_k`` and ``s_(k+1)`` agree to round-off,
      so any ``k``-dimensional slice of the run is as much the top ``k`` as any other.

    In either case LAPACK returns one arbitrary choice, and the cosines change with the thread
    count and the driver: a heavily quantized weight matrix, mostly zeros and a few repeated
    levels, read 0.345 to 0.349 at one ``k`` across five thread counts and two drivers. Round-off is
    ``eps * s_0 * sqrt(max(shape))``, the bound this package uses for the same question in
    ``transport``.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"principal_angles: k must be >= 1; got {k}")
    out = []
    for name, M in (("A", A), ("B", B)):
        F = as_frame(M, name=name)
        _, s, Vt = np.linalg.svd(F, full_matrices=False)
        if s.size < k:
            raise ValueError(f"principal_angles: k={k} exceeds the available rank of {name} "
                             f"({s.size})")
        limit = float(np.finfo(np.float64).eps * s[0] * np.sqrt(max(F.shape)))
        rank = int((s > limit).sum())
        if rank < k:
            raise ValueError(f"principal_angles: k={k} exceeds the numerical rank of {name} "
                             f"({rank} singular values above round-off {limit:.3g}): its top-{k} "
                             f"subspace is not defined")
        if k < s.size and s[k - 1] - s[k] <= limit:
            raise ValueError(f"principal_angles: k={k} splits a run of equal singular values of "
                             f"{name} (s_{k} - s_{k + 1} = {s[k - 1] - s[k]:.3g}, round-off "
                             f"{limit:.3g}): its top-{k} subspace is not defined")
        out.append(Vt[:k].T)
    Va, Vb = out
    return np.clip(np.linalg.svd(Va.T @ Vb, compute_uv=False), 0.0, 1.0)


def gram_spectrum(G) -> np.ndarray:
    """Descending singular values from a precomputed Gram matrix ``A.T @ A``.

    The Gram is half the cost of ``energy_spectrum`` at ``d = 2560`` -- 0.76 s of matmul against
    2.25 s of eigendecomposition -- and it is reusable in a way the spectrum is not. For a
    transport and its identity-free residual the two Grams are related exactly::

        (J - aI)^T (J - aI) = J^T J - a (J + J^T) + a^2 I

    so the second costs ``O(d^2)`` instead of a second matmul: 0.28 s against 0.98 s measured.
    ``residual_gram`` builds it.
    """
    M = as_frame(G, name="G")
    ev = np.linalg.eigvalsh(M)
    return np.sqrt(np.clip(ev, 0.0, None))[::-1]


def residual_gram(gram, J, alpha: float) -> np.ndarray:
    """The Gram of ``J - alpha*I`` from the Gram of ``J``, without a second matmul.

    Exact, not an approximation -- it is the expansion of the product, and it agrees with the
    direct computation to 4.4e-16 at ``d = 2560``.
    """
    G = np.array(as_frame(gram, name="gram"), copy=True)
    A = as_frame(J, name="J")
    a = float(alpha)
    G -= a * (A + A.T)
    G[np.diag_indices(G.shape[0])] += a * a
    return G
