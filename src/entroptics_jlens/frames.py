"""Boundary conversion: anything the caller hands in becomes a float64 numpy 2-D array,
or the call refuses.

The Jacobian lens ships its transports as float16 (``lens.py``'s ``save`` casts for storage).
Every read downstream is spectral -- a de-biased per-cell variance, a Tracy-Widom deviate, a
pseudo-inverse -- and none of them survives half precision, so the upcast happens once, here,
at the boundary.  Nothing in this package accepts a lower-precision array silently.
"""
from __future__ import annotations

import numpy as np


class FrameError(ValueError):
    """A surface that cannot be read as a 2-D float64 frame."""


def as_frame(X, *, name: str = "frame") -> np.ndarray:
    """``X`` (numpy array, torch tensor, or anything array-like) as a 2-D float64 numpy array.

    Refuses rather than repairing: a non-2-D shape, an empty axis, or a non-finite cell is a
    broken input, and a read taken on a silently patched frame is not a measurement.
    """
    A = X
    if hasattr(A, "detach"):                       # a torch tensor, on any device
        A = A.detach().to("cpu")
        # bfloat16 has no numpy counterpart, so the dtype conversion has to happen on the torch
        # side -- `.numpy()` on a bf16 tensor raises "unsupported ScalarType BFloat16", and a
        # published checkpoint has already turned up in bf16 once. A complex tensor is passed
        # through as complex instead, so the single check below refuses it rather than a second
        # copy of that message living here.
        A = A.numpy() if _torch_is_complex(A) else A.to(dtype=_torch_f64()).numpy()
    A = np.asarray(A)
    # Checked BEFORE the cast, because the cast is what destroys the evidence. `np.asarray(z,
    # dtype=float64)` on a complex array drops the imaginary part and raises only a numpy
    # ComplexWarning, which this package does not turn into an error. Every read downstream is
    # spectral, and the spectrum of the real part is not the spectrum of the matrix, so a
    # complex input is refused rather than silently halved.
    if np.iscomplexobj(A):
        imag = float(np.abs(A.imag).max())
        raise FrameError(
            f"{name}: complex input (largest |imaginary part| {imag:.6g}). Casting it to float64 "
            f"would discard the imaginary part and every spectral read here would then be taken "
            f"on a different matrix than the one supplied. Pass the real frame you mean -- "
            f"`X.real`, `np.abs(X)`, or the real-valued construction the complex one came from.")
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise FrameError(f"{name}: expected a 2-D (rows, cols) frame; got shape {A.shape}")
    if min(A.shape) == 0:
        raise FrameError(f"{name}: expected both axes non-empty; got shape {A.shape}")
    if not np.isfinite(A).all():
        bad = int((~np.isfinite(A)).sum())
        raise FrameError(f"{name}: {bad} of {A.size} cells are not finite; a spectral read of a "
                         f"frame with NaN or inf is undefined, and imputing them would invent data")
    return A


def _torch_f64():
    import torch
    return torch.float64


def _torch_is_complex(t) -> bool:
    import torch
    return bool(torch.is_complex(t))
