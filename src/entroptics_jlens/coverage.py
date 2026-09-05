"""Coverage: how much of a signal's resolved subspace a readout actually spans.

This replaces the complement read, which measured the wrong thing. ``Screen.uncondensed``
removes the receiver's directions and then re-reads the residual frame from scratch, and that
re-read whitens per channel. Whitening varies under projection -- removing a dominant direction
rescales every channel and can lift other modes above the floor -- so the mode count after
projection falls by some other amount than the number of directions removed. Measured: it failed an
analytic benchmark where the overlap was exact by construction, and returned the same answer for
a random matrix as for the real one.

The quantity actually wanted is a subspace overlap, and it has a standard form:

    coverage = || V_s^T V_t ||_F^2 / k_s

with ``V_s`` an orthonormal basis for the signal's resolved subspace (``k_s`` directions) and
``V_t`` one for the readout's (``k_t``). This is the sum of squared canonical correlations
between the two subspaces, normalised to ``[0, 1]``: it is ``1`` when the signal's subspace lies
inside the readout's, and ``0`` when they are orthogonal.

**And it has an analytic null.** For a ``k_t``-dimensional subspace drawn uniformly at random in
``R^d``, the expected squared projection of any fixed unit vector onto it is ``k_t / d``, so

    E[coverage under the null] = k_t / d

That baseline is what the old read lacked. At ``d = 768`` with a readout spanning ~20 directions
the null coverage is ~2.6%, which is to say ~97% of a signal lies "outside" a random readout by
chance alone. Reporting that as a finding is what went wrong; reporting coverage against its null
is what fixes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import as_frame


@dataclass(frozen=True)
class Coverage:
    """How much of the signal's resolved subspace the readout spans, against chance."""
    coverage:   float          # ||V_s^T V_t||_F^2 / k_s, in [0, 1]
    null:       float          # k_t / d, the random-subspace expectation
    excess:     float          # (coverage - null) / (1 - null): 0 at chance, 1 at full span
    k_signal:   int
    k_readout:  int
    d:          int
    cosines:    np.ndarray     # canonical correlations, descending

    # Calibrated scope, measured in `research/experiments/exp19_recalibrate.py` against a constructed
    # ground truth (centred cosine to the pre-norm final residual direction):
    #
    #     rank truncation   ordered correctly at a true gap of 0.07%, 100% of layers
    #     added noise       ordered correctly at 1.04%, 100% of layers; 88% from 0.09%
    #
    # There is deliberately no `resolves_gap(other)` helper here: the reliability of a
    # comparison is **not recoverable from the coverage values themselves**.
    # Across the calibrated degradations, separations below 0.001 in `excess` still order
    # correctly 89% of the time (n=44, p<0.0001). The catalogue's two Qwen3.5-4B fits sit at those
    # same separations and order *backwards*, consistently, because they differ by fit sample size
    # -- a mode outside the calibration. Same separation, opposite
    # reliability, so a threshold on it conflates the two situations. What governs is the KIND of
    # difference between the two readouts, which the caller knows and this object never sees.

    @property
    def above_chance(self) -> bool:
        """Is the overlap larger than a random readout of the same size would give?

        The threshold is deliberately crude -- twice the null -- because the point is to catch
        the case the old read missed, where the answer *is* chance. A calibrated test needs the
        null's variance, which ``coverage_null_sample`` supplies empirically.
        """
        return self.coverage > 2.0 * self.null


def _resolved_basis(W, far: float = 0.05):
    """An orthonormal basis for the directions ``W`` resolves above its own noise floor."""
    from entroptics.projection import noise_floor

    A = as_frame(W, name="frame")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((s > float(noise_floor(A, far=far, s=s))).sum())
    return Vt[:k].T, k


def coverage(signal, readout_image, *, far: float = 0.05) -> Coverage:
    """What fraction of ``signal``'s resolved subspace is spanned by ``readout_image``'s.

    ``signal`` is the frame as it stands (a residual stream); ``readout_image`` is the same frame
    after the readout has acted on it (the transported frame). Both are ``(T, d)`` on one basis.

    Returns the overlap, the chance level for a readout of that size, and the excess over
    chance. Read the excess: the raw coverage of a small readout in a wide space is near zero
    whatever it does, and that fact is a statement about the dimensions rather than the readout.
    """
    S = as_frame(signal, name="signal")
    R = as_frame(readout_image, name="readout image")
    if S.shape[1] != R.shape[1]:
        raise ValueError(f"signal has {S.shape[1]} columns and the readout image {R.shape[1]}; "
                         f"a subspace overlap needs both on one basis")
    d = int(S.shape[1])
    Vs, ks = _resolved_basis(S, far=far)
    Vt, kt = _resolved_basis(R, far=far)
    if ks == 0 or kt == 0:
        return Coverage(0.0, 0.0, 0.0, ks, kt, d, np.zeros(0))
    c = np.linalg.svd(Vs.T @ Vt, compute_uv=False)
    c = np.clip(c, 0.0, 1.0)
    cov = float((c ** 2).sum() / ks)
    null = float(kt / d)
    return Coverage(coverage=cov, null=null,
                    excess=float((cov - null) / (1.0 - null)) if null < 1.0 else 0.0,
                    k_signal=ks, k_readout=kt, d=d, cosines=c)


def coverage_null_sample(signal, k_readout: int, *, draws: int = 64, seed: int = 0,
                         far: float = 0.05) -> np.ndarray:
    """The null distribution, sampled: coverage of ``signal`` by random ``k_readout``-subspaces.

    The analytic mean is ``k_readout / d``; this gives its spread, which is what a decision
    needs. Use it to say "the readout covers more than 95% of random subspaces of the
    same size", which is a statement with a false-alarm level.
    """
    S = as_frame(signal, name="signal")
    d = int(S.shape[1])
    Vs, ks = _resolved_basis(S, far=far)
    if ks == 0:
        return np.zeros(int(draws))
    rng = np.random.default_rng(seed)
    out = np.empty(int(draws))
    for i in range(int(draws)):
        Q, _ = np.linalg.qr(rng.standard_normal((d, int(k_readout))))
        c = np.clip(np.linalg.svd(Vs.T @ Q, compute_uv=False), 0.0, 1.0)
        out[i] = float((c ** 2).sum() / ks)
    return out
