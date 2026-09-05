"""The Jacobian lens as an Entroptics ``Lens``.

A ``Lens`` is exactly its conversion: ``entry`` (surface -> the shared basis) and ``inverse``
(back out).  The Jacobian lens transports a residual-stream vector by ``h @ J.T``, which lands
in the final-layer residual basis -- the same ``d_model`` the stream itself lives in.  That is
what makes the screen well-posed: both sides place ``D = d_model``, and the pseudo-inverse of a
rank-``K`` truncation is a genuine inverse conversion, so ``certify``, ``realise`` and
``linear`` all mean something.

The vocabulary readout is the OTHER basis (``D = |vocabulary|``) and therefore the other screen.
It is registered entry-only: there is no honest inverse from logits back to a residual, and
declaring one would put a fabricated conversion where a measurement belongs.

Every line of domain code in this package lives in this module and in ``io``.  The library
stays domain-agnostic; nothing here is proposed for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from entroptics import Projection, Screen

from .frames import as_frame
from .transport import TransportSpectrum, resolved_transport, transport_spectrum

STREAM = "stream"
TRANSPORT = "jlens"


def stream_side() -> dict:
    """The raw residual stream: the identity conversion.  It is a side of the screen because
    the question is what the transport does RELATIVE to it."""
    return dict(entry=lambda H: as_frame(H, name="stream surface"))


def truncated_pair(J, rank: int):
    """A rank-``k`` truncation of ``J`` and its **exact** pseudo-inverse, from one SVD.

    Reaching for ``np.linalg.pinv`` on the reconstructed truncation is the obvious move and it is
    wrong. The reconstruction ``(U_k s_k) V_k^T`` carries float-noise singular values beyond rank
    ``k``; ``pinv``'s default cutoff is relative to the *largest* singular value, so on a transport
    whose spectrum spans many decades that noise survives the cut and is inverted into enormous
    values. The round trip then stops being a projection.

    Measured on gpt2, where the rank the participation ratio selects falls with depth::

        layer  rank   top sv of pinv(J_k) J_k   top sv of this pair
            5    23                   1.00000               1.00000
            6    17                   1.08224               1.00000
            9     3                   1.31455               1.00000

    An orthogonal projector has top singular value exactly 1, so 1.31 is an amplifying round trip,
    and ``certify`` read a residual above 1 from it -- an energy "share" of -7.8%.

    The cut is 1e-15 relative to the LARGEST singular value, and the reconstruction's noise lands
    just above it rather than far above: at gpt2 layer 9, rank 3, the leak reaches 4.7e-14 against
    a cutoff of 2.2e-14, and ``pinv(J_k)`` then reaches a top singular value of 4.4e13. Passing
    ``rcond=1e-10`` on the same matrix returns the round trip to exactly 1. That the margin is a
    factor of two, not of thousands, is why the failure is data-dependent and why a synthetic
    fixture does not reproduce it by default -- see ``tests/test_truncated_pair.py::lens_like``,
    which needs the transport's scale, its float16 storage and one massive-activation cell
    together before the naive route breaks at all, and then only mildly.

    Building the inverse from the same factors that built the truncation removes the failure by
    construction, and costs one SVD rather than two. Entry and inverse are a matched pair, the
    same rule ``vocab_side`` states for its own conversion.
    """
    A = as_frame(J, name="J")
    k = int(rank)
    if not 1 <= k <= min(A.shape):
        raise ValueError(f"rank {k} outside [1, {min(A.shape)}] for a {A.shape} transport")
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    # Refuse a rank that reaches the arithmetic's own noise. Inverting a singular value at the
    # float floor divides by numerical dust and amplifies it; the round trip stops contracting and
    # the "residual" that comes back is the arithmetic, not the transport. The bound is the
    # standard one for a computed SVD -- max(shape) * eps, relative to the largest value.
    floor = max(A.shape) * np.finfo(A.dtype).eps * float(sv[0])
    if sv[0] <= 0:
        raise ValueError("the transport is identically zero; it has no row space to project onto")
    if float(sv[k - 1]) <= floor:
        raise ValueError(
            f"rank {k} reaches singular value {float(sv[k - 1]):.3e}, at or below this "
            f"transport's arithmetic floor {floor:.3e}. Inverting it amplifies round-off rather "
            f"than inverting the transport. Truncate lower, or read the saturation as the "
            f"finding.")
    J_K = (U[:, :k] * sv[:k]) @ Vt[:k]
    J_pinv = (Vt[:k].T / sv[:k]) @ U[:, :k].T
    return J_K, J_pinv


def transport_side(J_K, pinv=None) -> dict:
    """The transport and its pseudo-inverse.

    ``entry(H) = H @ J_K.T`` is the Jacobian lens's own transport (``jlens/lens.py`` applies it
    exactly this way).  ``inverse(C) = C @ pinv(J_K).T``, so the round trip
    ``inverse(entry(H)) = H @ (pinv(J_K) J_K).T`` is ``H`` projected onto the resolved row
    space -- which is why ``Screen.certify`` on this side measures precisely the component of
    the residual stream the transport annihilates.

    ``pinv`` is the precomputed pseudo-inverse. It is a full SVD of a ``d x d`` matrix -- tens
    of seconds at ``d = 2560`` -- and it depends only on ``J_K``, so a caller reading many
    streams through one transport should compute it once and pass it in rather than paying for
    it per stream.
    """
    Jk = as_frame(J_K, name="J_K")
    Jp = np.linalg.pinv(Jk) if pinv is None else as_frame(pinv, name="pinv")
    return dict(entry=lambda H: as_frame(H, name="transport surface") @ Jk.T,
                inverse=lambda C: as_frame(C, name="concept frame") @ Jp.T)


def vocab_side(J_K, head, keep, *, norm_gain=None, norm_eps: float | None = None,
               invertible: bool = False) -> dict:
    """The readout to a token sub-basis -- the conversion that lets two models meet.

    Models of different residual width share no residual basis, but if their token ids agree they
    share a readout basis, so the vocabulary is where a crossing can happen at all.

    ``head`` is the output-head matrix and ``keep`` the token ids to land on. ``keep`` is
    required: the full vocabulary is ~1e5 columns against ~1e3 rows, where a per-channel
    whitening has nothing to estimate.

    ``norm_gain`` is the receiving model's final-norm gain and ``norm_eps`` its RMSNorm
    epsilon; both belong to the model, and supplying the gain without the epsilon is refused.
    Supply them and ``entry`` applies the model's actual readout, ``RMSNorm(x) * g @ W_k^T``; omit it and ``entry`` is the linear map
    ``x @ J_K^T @ W_k^T``. **Entry and inverse are always a matched pair**: an inverse that
    undoes a step the entry never applied is not an inverse, so the signature takes both
    together.

    ``invertible`` adds the ``inverse``, making the side two-way. What it recovers depends on
    which entry is in force:

      * without ``norm_gain`` the conversion is linear and the inverse is exact, up to the
        conditioning of the head;
      * with ``norm_gain`` the entry contains ``RMSNorm``, which is scale-invariant **by
        construction**, so a row's magnitude is not recoverable from its output. The direction is.

    Either way the head inverts by least squares -- solving ``L = Z W_k^T`` for ``Z`` is well
    posed whenever ``len(keep) >= d`` and ``W_k`` has full column rank -- and ``Screen.certify``
    measures whatever the round trip actually costs rather than assuming it away. For carrying a
    steering vector or a feature between models the direction is the payload and the scale is a
    coefficient the caller sets.
    """
    Jk = as_frame(J_K, name="J_K")
    idx = np.asarray(keep, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("keep: a 1-D, non-empty index of token ids is required")
    Wk = as_frame(head, name="head")[idx]
    g = None if norm_gain is None else np.asarray(norm_gain, dtype=np.float64).ravel()
    if g is not None and norm_eps is None:
        raise ValueError(
            "norm_gain was supplied without norm_eps. The entry then applies an RMSNorm, and the "
            "epsilon belongs to the receiving model (config.rms_norm_eps) rather than to this "
            "function; a house value normalises by something the model never used.")
    if g is not None and g.size != Jk.shape[0]:
        raise ValueError(f"norm_gain has {g.size} entries but the transport is {Jk.shape[0]} "
                         f"wide; these are not the same model's readout")

    # The model's own RMSNorm epsilon, never a house value. A crossing that normalises with a
    # different epsilon than the receiver applies is undoing an operation nobody performed, and
    # the constant is in the model's config: Qwen 1e-6, Llama 1e-5, and they differ by family.
    # Required whenever `norm_gain` is supplied, because that is when the entry contains the norm.
    eps = float(norm_eps) if norm_eps is not None else None

    def _rms(X):
        return X / np.sqrt((X ** 2).mean(1, keepdims=True) + eps)

    def _entry(H):
        X = as_frame(H, name="vocab surface") @ Jk.T
        return (_rms(X) * g if g is not None else X) @ Wk.T

    side = dict(entry=_entry)
    if not invertible:
        return side
    if Wk.shape[0] < Wk.shape[1]:
        raise ValueError(
            f"invertible: the head restricted to keep is {Wk.shape}; a least-squares inverse "
            f"needs at least as many kept tokens as residual dimensions, else the residual is "
            f"underdetermined and the returned vector would be one of infinitely many")
    # `np.linalg.pinv` carries a chosen cutoff -- rcond relative to the largest singular value --
    # and on a spectrum spanning decades it discards genuine directions or keeps float noise and
    # inverts it into enormous values. Measured elsewhere on these transports: it turned a round
    # trip that must contract into one with top singular value 1.31. The crossing is the last
    # place to accept an arbitrary constant, so both inverses take a cutoff derived from the
    # arithmetic instead: max(shape) * eps, the backward-error bound for a computed SVD.
    def _exact_pinv(A, name):
        U, sv, Vt = np.linalg.svd(A, full_matrices=False)
        if sv.size == 0 or sv[0] <= 0:
            raise ValueError(f"{name} is identically zero; it has no inverse conversion")
        keep_sv = sv > max(A.shape) * np.finfo(A.dtype).eps * sv[0]
        k = int(keep_sv.sum())
        if k == 0:
            raise ValueError(f"{name} has no singular value above its own arithmetic floor")
        return (Vt[:k].T / sv[:k]) @ U[:, :k].T

    Wp, Jp = _exact_pinv(Wk, "the head restricted to keep"), _exact_pinv(Jk, "the transport")

    def _inverse(L):
        Z = as_frame(L, name="vocab frame") @ Wp.T
        if g is not None:
            Z = Z / np.where(np.abs(g) > 0, g, 1.0)
            n = np.linalg.norm(Z, axis=1, keepdims=True)      # RMSNorm ate the scale
            Z = Z / np.where(n > 0, n, 1.0)
        return Z @ Jp.T

    side["inverse"] = _inverse
    return side


@dataclass(frozen=True)
class LayerScreen:
    """A layer's stream/transport screen, with the spectrum that fixed the truncation."""
    screen:    Screen
    spectrum:  TransportSpectrum
    layer:     int
    J_K:       np.ndarray


def layer_screen(H, J, *, layer: int, far: float = 0.05, seed: int = 0,
                 rank: str | int | None = None, pinv=None,
                 spectrum: TransportSpectrum | None = None) -> LayerScreen:
    """Place the residual stream at ``layer`` on both sides of one screen: raw, and transported.

    ``H`` is ``(T positions, d_model)`` -- the ordered axis is token position within one prompt,
    which is what makes the ordered-axis reads (``coherence``, ``rates``) legitimate here, and
    what makes the noise floor legitimate on the STREAM even though it is not on ``J``: a stream
    is one sample, where a transport is a corpus average with its noise already removed.

    ``rank`` chooses the conversion:

      ``None`` (default)   use ``J`` whole. The complement read does not need a truncation --
                           ``uncondensed`` restricts to the directions outside what the RECEIVER
                           resolves, and the receiver's directions are read from the transported
                           frame against the screen's own floor, which is where its rank comes from. This is
                           the right default now that the resolved rank of a transport is known
                           to be unreliable (the Tracy-Widom edge presumes a noise bulk a
                           corpus-averaged Jacobian does not have; see ``nulls``).
      ``"resolved"``       truncate at ``transport_spectrum``'s ``K``. Gives ``certify`` a real
                           null space to measure, at the cost of depending on that floor.
      ``int``              truncate at an explicit rank, e.g. one taken from the participation
                           ratio, which needs no floor at all.

    Refuses a saturated transport under ``"resolved"``: if ``J`` resolves every mode its
    truncation is the identity, the null space is empty, and ``certify`` would report a
    losslessness that says nothing. That is a finding to report rather than route around.
    """
    Hf = as_frame(H, name="H")
    if rank is None:
        J_K = as_frame(J, name="J")
        # Both of these are d x d spectral work that depends only on the transport, never on
        # the stream. A caller reading many streams through one transport passes them in; at
        # d = 2560 recomputing them per stream dominates the whole run.
        spec = transport_spectrum(J_K, far=far) if spectrum is None else spectrum
    elif rank == "resolved":
        J_K, spec = resolved_transport(J, far=far)
        if spec.saturated:
            raise ValueError(
                f"layer {layer}: the transport resolves all {min(spec.shape)} modes at "
                f"far={far}, so the rank-K truncation is the identity and it has no null space. "
                f"certify() and uncondensed() on this layer would measure nothing. Report the "
                f"saturation; do not read a complement that does not exist.")
    elif isinstance(rank, int):
        A = as_frame(J, name="J")
        if not 1 <= rank <= min(A.shape):
            raise ValueError(f"layer {layer}: rank {rank} outside [1, {min(A.shape)}]")
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        J_K = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        spec = transport_spectrum(A, far=far)
    else:
        raise ValueError(f"rank must be None, 'resolved' or an int; got {rank!r}")
    if Hf.shape[1] != J_K.shape[1]:
        raise ValueError(f"layer {layer}: H has {Hf.shape[1]} columns and J takes "
                         f"{J_K.shape[1]}; the stream and the transport disagree on d_model")
    s_ = Screen(far=far, seed=seed)
    s_.register(STREAM, **stream_side())
    s_.register(TRANSPORT, **transport_side(J_K, pinv=pinv))
    s_.place(STREAM, Hf)
    s_.place(TRANSPORT, Hf)
    return LayerScreen(screen=s_, spectrum=spec, layer=int(layer), J_K=J_K)


def complement(ls: LayerScreen) -> Projection:
    """What the transport does not resolve, read as a projection on its own screen.

    ``Screen.uncondensed(stream, jlens)`` is the sending frame restricted to the directions
    outside what the receiver resolves, and ``K_signal`` counts the modes standing above the
    noise floor there.

    **Read the scope before using this number.** It was once this package's headline read, and
    the headline was wrong. It reported that 92-102% of a residual stream's resolved structure
    lies outside the transport, stable across six models in four families -- and a
    matched-spectrum surrogate and a random rank-K matrix return the same ~100%. In 2560
    dimensions two unaligned low-dimensional subspaces are near-orthogonal by default, so the
    read was returning its own null and the replication was the stability of that null
    (benchmark B3). Benchmark B2, where the overlap is exact by construction, is kept as a
    strict xfail so the boundary stays visible.

    The mechanism is that ``uncondensed`` re-reads the residual frame from scratch after removing
    the receiver's directions, and that re-read whitens per channel; whitening varies under
    projection, so the mode count falls by some other amount than the number of directions taken
    out.

    What it can still do: fire on a LARGE overlap. gemma-3-1b's early layers read 59.7%, roughly
    40 points below the null band, and there the complement removes exactly the number of
    directions the transport carries. Trust it to detect a big overlap, never to quantify one.

    **Use ``coverage`` instead**, which measures the same intent -- how much of the signal the
    readout spans -- as a subspace overlap with an analytic chance level.
    """
    rest = ls.screen.uncondensed(STREAM, TRANSPORT)
    return Projection(rest, far=ls.spectrum.far)
