"""Reading and writing the workspace at inference time.

Everything else in this package reads a matrix offline. This is the runtime half: the transport's
SVD is computed ONCE per model, and what happens per token is a projection onto the basis it
yields. Measured on gpt2 layer 6 (``d = 768``, ``k = 48``):

    preprocessing, once per model          479 ms
    extract, one token                    8.68 us
    extract, 512 tokens                    1.20 us/token
    inject, one token                     4.48 us
    the model's own forward pass        2241.7 us/token

**Extract is 0.054% of the forward pass.** The cost people assume is prohibitive here is the SVD,
and the SVD does not happen at runtime; it happens when you load the model. What is left is a
``(d, k)`` matvec.

Two operations:

  ``extract(h)``   where this token sits in the workspace -- ``k`` coordinates, one per resolved
                   direction of ``J - alpha*I``. A running read, cheap enough to take on every
                   token of every request.
  ``inject(h, j, amount)``  add ``amount`` along workspace direction ``j`` and hand back a
                   residual vector. The basis is orthonormal, so this moves exactly that
                   coordinate and leaves the other ``k - 1`` untouched -- asserted in
                   ``tests/test_workspace.py``, because a write that perturbs everything is a
                   different experiment from a write that moves one thing.

**Where to inject, and the measurement that decides it.** A computed vector written at the
EMBEDDING reproduces a real token 100% of the time on a copy task; the same vector written four
blocks later reproduces it 0% of the time. Depth is where writes
stop carrying. ``inject`` returns a vector and does not choose a site; the caller does, and the
caller should know that number.

**What this does NOT come with.** Five independent attempts to turn a read from this package into
a decision have failed. Extract is fast and exact -- it tells you the
coordinates. Whether a coordinate crossing some value means anything about the answer the model is
about to give is not established here, and a threshold on one is a hypothesis to test, not a
feature to ship.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decompose import decompose
from .frames import as_frame
from .transport import transport_spectrum


@dataclass(frozen=True)
class Workspace:
    """A model's workspace basis, precomputed. Build once, use per token.

    ``basis`` is ``(d, k)`` with orthonormal columns: the right singular vectors of
    ``J - alpha*I`` for the ``k`` directions it resolves. ``image`` is the corresponding ``(d, k)``
    left basis, which is where those directions LAND in the final-layer frame.
    """
    layer: int
    d: int
    k: int
    alpha: float
    identity_energy: float
    basis: np.ndarray          # (d, k) orthonormal: directions in the layer's own frame
    image: np.ndarray          # (d, k) orthonormal: where they land at the readout
    singular: np.ndarray       # (k,) the strength of each direction

    def extract(self, h) -> np.ndarray:
        """Workspace coordinates of one token or a batch of them.

        ``h`` is ``(d,)`` or ``(T, d)``; returns ``(k,)`` or ``(T, k)``. This is the runtime read
        and it is a matvec -- no SVD, no allocation beyond the result.
        """
        x = np.asarray(h, dtype=np.float64)
        if x.shape[-1] != self.d:
            raise ValueError(
                f"workspace is d={self.d} and the residual is {x.shape[-1]} wide; a projection "
                f"needs both on one basis. Check the layer this workspace was built for.")
        return x @ self.basis

    def inject(self, h, direction: int, amount: float) -> np.ndarray:
        """``h`` moved by ``amount`` along one workspace direction.

        Returns a new array; nothing is modified in place, because an in-place write into a
        model's activation is the kind of thing that is easy to leave switched on.
        """
        x = np.asarray(h, dtype=np.float64)
        if x.shape[-1] != self.d:
            raise ValueError(f"workspace is d={self.d} and the residual is {x.shape[-1]} wide")
        if not 0 <= int(direction) < self.k:
            raise ValueError(
                f"direction {direction} outside [0, {self.k}) -- this workspace resolves {self.k} "
                f"directions at layer {self.layer}. Directions are ordered by strength, so 0 is "
                f"the strongest.")
        return x + float(amount) * self.basis[:, int(direction)]

    def reconstruct(self, coords) -> np.ndarray:
        """Workspace coordinates back to a residual vector: the inverse of ``extract`` on the
        subspace. ``extract(reconstruct(c)) == c`` exactly, because the basis is orthonormal."""
        c = np.asarray(coords, dtype=np.float64)
        if c.shape[-1] != self.k:
            raise ValueError(f"this workspace has {self.k} directions and got {c.shape[-1]}")
        return c @ self.basis.T


def workspace(J, *, layer: int = -1, k: int | None = None, far: float = 0.05) -> Workspace:
    """Build the runtime workspace for one transport. Do this once, at load time.

    ``k`` defaults to the rank the identity-free transport resolves under the ``mp`` null at
    ``far``. Pass an integer to fix it instead -- a serving system usually wants a fixed width so
    the coordinate vector has a stable shape across model versions.

    The identity comes off first and that is not optional here. On raw ``J`` the leading singular
    directions at depth are the identity's, so a "workspace coordinate" would largely be reading
    the residual stream passing through: at Qwen3.5-4B layer 30, 79% of the matrix is that
    pass-through and the resolved rank reads 25 against 183 once it is removed.
    """
    A = as_frame(J, name="J")
    dec = decompose(A, kind="identity")
    M = dec.residual
    s_exact = np.linalg.svd(M, compute_uv=False)
    resolved = transport_spectrum(M, far=far, null="mp", s=s_exact).K
    width = int(resolved if k is None else k)
    if width < 1:
        raise ValueError(
            f"this transport resolves {resolved} directions above its own noise floor at "
            f"far={far}, so there is no workspace to read. Pass k explicitly to force a width, "
            f"knowing the extra directions are not distinguishable from noise.")
    if width > min(M.shape):
        raise ValueError(f"k={width} exceeds the transport's {min(M.shape)} directions")
    u, sv, vt = np.linalg.svd(M, full_matrices=False)
    return Workspace(layer=int(layer), d=int(M.shape[0]), k=width, alpha=dec.alpha,
                     identity_energy=dec.removed_energy, basis=np.ascontiguousarray(vt[:width].T),
                     image=np.ascontiguousarray(u[:, :width]), singular=sv[:width])
