"""Removing what the architecture puts there, before asking what the transport carries.

A residual stream adds ``h_l`` to everything downstream, so ``J_l = E[d h_final / d h_l]``
carries an identity component by construction, and it grows with depth as less transformation
remains above the layer. Measured:

    Qwen3.5-4B   layer  0   diag mean 0.181   off-diag rms 0.038    ||aI||/||J|| = 0.089
                 layer 12   diag mean 0.280   off-diag rms 0.019    ||aI||/||J|| = 0.265
                 layer 24   diag mean 1.037   off-diag rms 0.017    ||aI||/||J|| = 0.764
                 layer 30   diag mean 0.994   off-diag rms 0.010    ||aI||/||J|| = 0.889

At layer 30 the transport is 89% identity by Frobenius norm and its MEDIAN singular value is
1.05 -- which is alpha. The flat block a spectral floor reads as "the bulk" is the identity
itself. Removing it moves the resolved count from 25 to 183 at that layer, and reverses the
apparent decline of ``K`` with depth: the decline was the identity growing, flattening the
spectrum and burying everything under its own estimated variance.

Nothing here is fitted. ``alpha = tr(J)/d`` is exactly the orthogonal projection of ``J`` onto
``span(I)`` under the Frobenius inner product -- the unique least-squares coefficient, against a
basis element the architecture guarantees is present.

Two decompositions, and they answer different questions:

  ``identity``  remove ``alpha I``, one coefficient. "What does the transport do beyond passing
                the stream through uniformly?"
  ``diagonal``  remove ``diag(J)``, d coefficients. "What does it do beyond per-coordinate
                gain?" Stronger, and appropriate when the diagonal is not uniform -- at Qwen
                layer 0 the diagonal's sd (0.607) exceeds its mean (0.181), so a single alpha
                does not describe it.

Report which one a number was read under. They are different measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import as_frame


@dataclass(frozen=True)
class Decomposition:
    """A transport split into what the architecture supplies and what is left."""
    kind:            str
    alpha:           float        # tr(J)/d; the identity coefficient (nan for "diagonal")
    residual:        np.ndarray   # J minus the removed component
    removed_energy:  float        # ||removed||_F^2 / ||J||_F^2
    diag_mean:       float
    diag_sd:         float
    offdiag_rms:     float

    @property
    def identity_dominated(self) -> bool:
        """More of the transport is pass-through than is transformation. A spectral floor read
        on the undecomposed matrix is then reading the identity's flat block as its bulk."""
        return self.removed_energy > 0.5


def _stats(A: np.ndarray) -> tuple[float, float, float]:
    d = min(A.shape)
    dg = np.diag(A)
    off_sq = float((A ** 2).sum() - (dg ** 2).sum())
    n_off = A.size - d
    return (float(dg.mean()), float(dg.std()),
            float(np.sqrt(off_sq / n_off)) if n_off > 0 else 0.0)


def decompose(J, *, kind: str = "identity") -> Decomposition:
    """Split ``J`` into the architectural component and the residual transport.

    ``kind="identity"`` removes ``alpha I`` with ``alpha = tr(J)/d``; ``kind="diagonal"``
    removes the full diagonal. Square matrices only -- a transport maps the residual basis to
    itself, and a non-square input is a different object that this decomposition does not
    describe.
    """
    A = as_frame(J, name="J")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"decompose expects a square transport; got {A.shape}. The identity "
                         f"component is only defined for a map from a basis to itself.")
    d = A.shape[0]
    total = float((A ** 2).sum())
    dm, ds, off = _stats(A)
    if kind == "identity":
        alpha = float(np.trace(A) / d)              # = <J, I>/<I, I>: the exact projection
        removed_sq = alpha ** 2 * d
        R = A - alpha * np.eye(d)
    elif kind == "diagonal":
        alpha = float("nan")
        dg = np.diag(A)
        removed_sq = float((dg ** 2).sum())
        R = A - np.diag(dg)
    else:
        raise ValueError(f"unknown decomposition {kind!r}; expected 'identity' or 'diagonal'")
    return Decomposition(kind=kind, alpha=alpha, residual=R,
                         removed_energy=(removed_sq / total) if total > 0 else 0.0,
                         diag_mean=dm, diag_sd=ds, offdiag_rms=off)


#: Below this identity share the change measured across the swept lenses is 1.0x -- reading the
#: raw transport gives the same answer. Every layer whose change exceeded 1.5x had a share
#: between 0.348 and 0.778; every layer where it did not sat between 0.003 and 0.433. The two
#: populations meet here. It is a screening threshold and nothing downstream depends on it: `screen`
#: reports the share itself, and a caller who wants a different cut has the number.
WORTH_DECOMPOSING = 0.4


def identity_share(J) -> tuple[float, float]:
    """``(alpha, identity_energy)`` without an SVD.

    ``alpha = tr(J)/d`` costs a diagonal read and the share costs one Frobenius norm, so this is
    the whole identity measurement at ``O(d^2)`` against the ``O(d^3)`` of a spectrum. Measured at
    ``d = 2560``: **130 ms against 3047 ms**, a 23x saving, and the arithmetic is identical to
    ``decompose`` -- it is the same two numbers, without building ``J - alpha*I``.

    Use it to decide whether a layer is worth the SVD. The share is what predicts the change:
    below about 0.4 the two reads agree.
    """
    A = as_frame(J, name="J")
    d = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"identity_share expects a square transport; got {A.shape}")
    alpha = float(np.trace(A)) / d
    flat = A.reshape(-1)
    total = float(flat.dot(flat))
    return alpha, ((alpha * alpha * d) / total) if total > 0.0 else 0.0


def screen(lens, layers=None, *, threshold: float = WORTH_DECOMPOSING) -> list[dict]:
    """Every layer's identity share, with no SVD anywhere. The sweep before the expensive read.

    Returns one row per layer: ``layer``, ``alpha``, ``identity``, and ``worth_decomposing``.
    A caller then pays for a spectrum only on the layers that flagged, which on a 31-layer
    ``d = 2560`` lens is seconds instead of a minute and a half.

    ``lens`` is a ``LensFile`` or anything with ``jacobian(layer)`` and ``source_layers``.
    """
    picked = list(lens.source_layers if layers is None else layers)
    rows = []
    for layer in picked:
        alpha, share = identity_share(lens.jacobian(layer))
        rows.append({"layer": layer, "alpha": alpha, "identity": share,
                     "worth_decomposing": bool(share >= threshold)})
    return rows
