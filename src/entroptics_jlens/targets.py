"""The object a transport is scored against, and how to recover it.

Scoring a Jacobian lens against "the model's final residual" has two requirements, and a reading
that skips either one measures something else. They are in this module rather than in an
experiment script because any future comparison meets them again.

**The prediction is affine.** A Jacobian lens is a first-order expansion,

    h_final  ~=  h_final(h0)  +  J (h_l - h0)

so it carries an offset. Scoring the bare product ``|| F - H J^T ||`` omits it, and a fitted
scalar gain silently absorbs it -- measured on Qwen3.5-4B the best gain runs 1.9 to 4.9 and is
never near 1, which is the signature of a missing intercept rather than a property of the lens.
Centring over tokens removes it.

**The target is the PRE-norm residual.** ``hidden_states[-1]`` from HuggingFace arrives with the
model's final norm already applied: on Qwen3.5-4B its mean per-token norm is 156.1 against 52.1
for its neighbour, a 3x discontinuity that is exactly the RMSNorm. But ``J`` maps into the
un-normalised stream, which is why a readout applies the final norm itself. Predicting the
normalised vector with a map that outputs an unnormalised one inflates the error at the top of the
network and leaves the bottom alone.

The pre-norm residual is recoverable without re-running the model. RMSNorm is

    y = (x / rms(x)) * w        elementwise in w

so ``y / w`` is ``x`` at unit rms: the direction exactly, with only the per-token scale lost. That
needs one vector from the checkpoint rather than the weights. Direction is the right target anyway --
the unembedding is linear, so direction is what fixes the logits up to a scale.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np


def final_norm_weight(repo: str = "models--Qwen--Qwen3.5-4B",
                      key: str = "model.language_model.norm.weight") -> np.ndarray:
    """The final RMSNorm gain vector, read straight from a local checkpoint.

    Loaded through torch because the tensor is stored bf16, which numpy will not parse. The
    returned vector is float64; its bf16 provenance carries ~3 decimal digits, which is ample for
    a divisor whose only job is to undo a diagonal scaling.
    """
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:                                    # pragma: no cover - env
        raise ImportError(
            "reading a model's final-norm weight needs torch and safetensors (it comes from a "
            "HuggingFace snapshot): pip install 'entroptics-jlens[lens]'") from exc

    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{repo}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {repo}")
    index = Path(snaps[0], "model.safetensors.index.json")
    if not index.exists():
        raise FileNotFoundError(f"no safetensors index at {index}")
    weight_map = json.loads(index.read_text())["weight_map"]
    if key not in weight_map:
        raise KeyError(f"{key!r} is not in {index}; the checkpoint names its final norm "
                       f"differently and the caller must say which")
    with safe_open(str(Path(snaps[0], weight_map[key])), framework="pt") as f:
        return f.get_tensor(key).to(torch.float64).numpy()


def prenorm_direction(post_norm, w) -> np.ndarray:
    """Undo a final RMSNorm: the pre-norm residual's direction, at unit rms.

    Exact up to the per-token scale, which the norm discarded and nothing can return. Refuses a
    gain vector with a zero entry rather than dividing by it, because the direction along that
    coordinate is genuinely gone, and a large finite number would stand in for it.
    """
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(post_norm, dtype=np.float64)
    if y.shape[-1] != w.size:
        raise ValueError(f"frame has {y.shape[-1]} columns and the gain vector {w.size}")
    if not np.all(np.abs(w) > 1e-12):
        raise ValueError("final norm gain has a zero entry; the pre-norm direction is not "
                         "recoverable along it")
    return y / w


def centred_cosine(predicted, target) -> float:
    """Agreement on the token-varying part: centre both over tokens, then per-token cosine.

    Centring is what makes this a measure of the transport rather than of the shared component.
    Both frames are dominated by a direction they agree on whatever the transport does, and an
    uncentred cosine reports mostly that.

    Equal to the correlation whose square is the explained variance of the token-varying part, so
    ``centred_cosine ** 2`` is directly comparable with an affine ``R^2``.
    """
    P = np.asarray(predicted, dtype=np.float64)
    X = np.asarray(target, dtype=np.float64)
    if P.shape != X.shape:
        raise ValueError(f"predicted {P.shape} and target {X.shape} must match")
    Pc = P - P.mean(0, keepdims=True)
    Xc = X - X.mean(0, keepdims=True)
    den = np.sqrt((Pc ** 2).sum(1) * (Xc ** 2).sum(1))
    return float(np.mean((Pc * Xc).sum(1) / np.maximum(den, 1e-300)))


def rms_normalize(frame, w) -> np.ndarray:
    """Apply an RMSNorm with gain ``w``, the operation the readout performs.

    This is the honest way to compare a transport with the model's final residual, and it is
    cheaper than the alternative: rather than undoing the norm on the target, apply it to the
    prediction, so both sides are the object the unembedding actually consumes.

    The model's logits are ``head(rms_normalize(x, w))``; a lens's are
    ``head(rms_normalize(J h, w))``. Comparing those two is comparing what is read, and it removes
    the per-token scale from both sides rather than from one -- which matters, because a
    token-centred statistic is *not* invariant under rescaling one side per token.
    """
    w = np.asarray(w, dtype=np.float64)
    X = np.asarray(frame, dtype=np.float64)
    if X.shape[-1] != w.size:
        raise ValueError(f"frame has {X.shape[-1]} columns and the gain vector {w.size}")
    return (X / np.sqrt((X ** 2).mean(-1, keepdims=True))) * w
