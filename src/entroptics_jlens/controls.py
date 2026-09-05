"""Entropy-matched nulls.  A read without one is a fit.

Each control destroys exactly one property of ``J`` and preserves the rest, so the read that
survives one control and dies on another names what it was measuring:

  ``gaussian_null``     iid Gaussian, same shape.  Calibration: the derived edge must return
                        ~``far`` false modes on pure noise.  If this reads a large ``K``, the
                        floor is wrong and nothing downstream means anything.
  ``shuffled_entries``  ``J``'s entries permuted.  The entry distribution is preserved
                        EXACTLY -- no Gaussian assumption -- and all structure is gone.  A
                        resolved rank that survives this is dimensionality.
                        **Apply it to ``J - alpha*I``, never to ``J``.**  The permutation moves
                        the diagonal off the diagonal, so it destroys the identity the residual
                        stream guarantees is present -- and the identity is exactly what lifts
                        the self-estimated floor on the real side.  Measured on gpt2 (mp,
                        far=0.05): on raw ``J`` the real transport resolves FEWER modes than its
                        own shuffle at layers 7, 8, 9 and 10 (layer 9: 6 against 34), which reads
                        as a paradox and is the identity.  On ``J - alpha*I`` the inversion is
                        gone at 7-9 (layer 9: 46 against 21, stable over 8 seeds) and survives
                        only at layer 10.
  ``matched_spectrum``  ``J``'s singular values with Haar-random orthogonal factors.  The
                        spectrum is preserved exactly, the alignment between ``J``'s subspaces
                        and the residual stream is destroyed.  This is the control for every
                        read taken THROUGH ``J`` (``certify``, the complement, ``transfer``):
                        if the matched-spectrum lens reads the same, the result is a fact about
                        ``J``'s singular values and not about where it points.

``rng`` is required, never defaulted: a control drawn from an unnamed seed cannot be re-run.

The caveat behind all of them
-----------------------------
The default ``mp`` floor estimates its per-cell variance from the very matrix it is judging, so
**anything that concentrates energy raises the bar the concentrated matrix then has to clear.**
Both surprises measured on gpt2 are that one mechanism:

  * an identity core concentrates energy on the diagonal -- layers 7-9 above, fixed by
    ``decompose``;
  * a near rank-one spectrum concentrates it in one direction -- gpt2 layer 10, where
    ``PR(J - alpha*I)`` is 1.5 and the real transport still resolves 39 against its shuffle's
    44-51 over 8 seeds. Decomposition does not touch this one, and nothing here fixes it.

So a control that removes structure will sometimes read HIGHER than the real matrix, and that is
not evidence the real matrix is noise. It is the floor moving. Compare like with like: decompose
first, and report ``PR`` beside ``K`` so a concentrated spectrum is visible rather than inferred.
"""
from __future__ import annotations

import numpy as np

from .frames import as_frame


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """An ``(n, n)`` orthogonal matrix, Haar-distributed (QR of a Gaussian, sign-corrected --
    without the sign correction the QR of a Gaussian is not Haar)."""
    Q, R = np.linalg.qr(rng.standard_normal((int(n), int(n))))
    return Q * np.sign(np.diag(R))


def gaussian_null(shape, rng: np.random.Generator, *, sigma: float = 1.0) -> np.ndarray:
    """iid ``N(0, sigma^2)``, given shape.

    ``sigma`` barely matters and is exposed only so a caller can match one: the noise floor
    estimates its per-cell variance from the screen it is handed, so it rescales with the
    matrix and the resolved count is scale-invariant.  That invariance is itself worth
    asserting, and ``tests/test_controls.py`` does."""
    N, F = (int(v) for v in shape)
    return float(sigma) * rng.standard_normal((N, F))


def shuffled_entries(J, rng: np.random.Generator) -> np.ndarray:
    """``J``'s entries, permuted over the whole matrix.  Same multiset of cells, no structure."""
    A = as_frame(J, name="J")
    return rng.permutation(A.reshape(-1)).reshape(A.shape)


def matched_spectrum(J, rng: np.random.Generator) -> np.ndarray:
    """``U_r diag(s) V_r^T`` for Haar-random ``U_r, V_r``: ``J``'s singular values exactly,
    pointing nowhere in particular."""
    A = as_frame(J, name="J")
    N, F = A.shape
    s = np.linalg.svd(A, compute_uv=False)
    k = s.size
    U = haar_orthogonal(N, rng)[:, :k]
    V = haar_orthogonal(F, rng)[:, :k]
    return (U * s) @ V.T


def frobenius_sigma(J) -> float:
    """``||J||_F / sqrt(N F)`` -- the per-cell rms, for matching a Gaussian null's scale."""
    A = as_frame(J, name="J")
    return float(np.sqrt((A ** 2).sum() / A.size))
