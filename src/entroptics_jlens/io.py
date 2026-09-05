"""Reading a fitted Jacobian lens off disk, one transport at a time.

``jlens/lens.py`` saves ``{"J": {layer: Tensor[d_model, d_model]}, "n_prompts",
"source_layers", "d_model"}``, with the transports cast to float16 for storage.

Two consequences shape this module:

  * **One layer at a time.**  A ~100-layer lens at ``d_model = 5120`` is ~5 GB in float16 and
    ~21 GB upcast.  ``LensFile`` holds the checkpoint (memory-mapped where the serialization
    allows it) and upcasts a single transport on request.  Nothing here materialises them all.
  * **Upcast at the boundary.**  float16 has ~3 decimal digits; a de-biased per-cell variance,
    a Tracy-Widom deviate and a pseudo-inverse all need more.  ``jacobian()`` returns float64
    and refuses anything it cannot convert (``frames.as_frame``).

Every failure here is a refusal with a message naming what was found.  A lens file that is
really a fitting checkpoint, a missing layer, a non-square transport: none of them has a
sensible default, and inventing one would put fabricated numbers in a report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .frames import as_frame


class LensFormatError(ValueError):
    """A checkpoint that is not a saved Jacobian lens."""


@dataclass
class LensFile:
    """A fitted lens on disk.  ``jacobian(layer)`` is the only way in."""
    path:          Path
    d_model:       int
    n_prompts:     int
    source_layers: list[int]
    mmapped:       bool
    _raw:          Any

    def jacobian(self, layer: int) -> np.ndarray:
        """Transport ``J_layer`` as a ``(d_model, d_model)`` float64 array."""
        J = self._raw["J"]
        key = layer if layer in J else str(layer)
        if key not in J:
            raise KeyError(f"{self.path}: no transport for layer {layer}; "
                           f"fitted layers are {self.source_layers}")
        A = as_frame(J[key], name=f"J[{layer}]")
        if A.shape != (self.d_model, self.d_model):
            raise LensFormatError(
                f"{self.path}: J[{layer}] has shape {A.shape}, expected "
                f"({self.d_model}, {self.d_model}) -- the transport maps the residual stream "
                f"into the final-layer basis, so it is square in d_model")
        return A

    def __len__(self) -> int:
        return len(self.source_layers)


def load_lens(path) -> LensFile:
    """Open a lens checkpoint saved by ``jlens.JacobianLens.save``.

    Refuses with a non-recoverable error if the file is absent, torch is not installed, or the
    checkpoint carries no ``"J"`` key (which is how ``jlens`` itself distinguishes a saved lens
    from a mid-fit checkpoint)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p}: no lens checkpoint. Fit one with jlens.fit(...).save(path), or download a "
            f"published lens.pt. This package does not synthesise a stand-in transport.")
    try:
        import torch
    except ImportError as exc:                                    # pragma: no cover - env
        raise ImportError(
            "reading a lens checkpoint needs torch (the file is a torch serialization): "
            "pip install 'entroptics-jlens[lens]'") from exc

    mmapped = True
    try:
        raw = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    except (RuntimeError, TypeError, ValueError):
        # Older / non-zipfile serializations cannot be mapped. The numbers are identical either
        # way; only the memory profile changes, so this is recorded on the lens.
        mmapped = False
        raw = torch.load(p, map_location="cpu", weights_only=True)

    if not isinstance(raw, dict) or "J" not in raw:
        keys = sorted(raw) if isinstance(raw, dict) else None
        found = (f"{keys[:12]} ({len(keys)} keys)" if keys is not None
                 else type(raw).__name__)   # say how many were elided, never just elide
        raise LensFormatError(
            f"{p}: no 'J' key, so this is not a saved lens (jlens treats the presence of 'J' as "
            f"what separates a lens from a fitting checkpoint). Found: {found}")

    J = raw["J"]
    layers = raw.get("source_layers")
    if layers is None:
        layers = sorted(int(k) for k in J)
    layers = [int(v) for v in layers]
    d_model = int(raw["d_model"]) if "d_model" in raw else int(as_frame(
        J[next(iter(J))], name="J[first]").shape[0])
    return LensFile(path=p, d_model=d_model, n_prompts=int(raw.get("n_prompts", 0)),
                    source_layers=layers, mmapped=mmapped, _raw=raw)
