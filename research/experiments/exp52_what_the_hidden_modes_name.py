"""What the modes the identity hides actually name, in tokens.

The claim in the paper is a count: at gpt2's deepest fitted layer a spectral read of ``J`` resolves
6 directions and a read of ``J - alpha*I`` resolves 39. A count says how many, not what they are,
and a reader is entitled to ask whether the extra 33 are structure or arithmetic.

This decodes them. A transport's left singular vectors live in the final residual basis, which is
the space the unembedding reads, so each one can be pushed through the model's own readout and
printed as the tokens it promotes and suppresses. Run it and the modes below the identity-free
floor name coherent things; the modes above it do not.

    python experiments/exp52_what_the_hidden_modes_name.py
    python experiments/exp52_what_the_hidden_modes_name.py --model gpt2 --top 10

Needs the gpt2 lens (``entroptics-jlens fetch gpt2``) and the gpt2 checkpoint from the HuggingFace
cache, for its unembedding and final-norm gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from entroptics_jlens.decompose import decompose            # noqa: E402
from entroptics_jlens.io import load_lens                   # noqa: E402
from entroptics_jlens.transport import transport_spectrum   # noqa: E402

LENS = "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"


def readout(model_id: str):
    """The model's own output head: the final-norm gain and the unembedding.

    Returned as float64 numpy. A direction decoded through anything else is a decode of that
    other thing.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:                                    # pragma: no cover - env
        raise ImportError(
            "decoding modes into tokens needs torch and transformers: "
            "pip install 'entroptics-jlens[lens]' transformers") from exc
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    W_U = model.lm_head.weight.detach().numpy().astype(np.float64)     # [vocab, d]
    g = model.transformer.ln_f.weight.detach().numpy().astype(np.float64)
    return tok, W_U, g


def name_mode(u: np.ndarray, W_U: np.ndarray, g: np.ndarray, tok, top: int):
    """The tokens one output direction promotes, and the ones it suppresses.

    A singular vector's sign is arbitrary, so both ends are reported and neither is "the" answer.
    The final-norm gain is applied because the readout applies it; skipping it decodes a vector
    the model never unembeds.
    """
    logits = W_U @ (u * g)
    order = np.argsort(logits)
    hi = [tok.decode([int(i)]) for i in order[::-1][:top]]
    lo = [tok.decode([int(i)]) for i in order[:top]]
    return hi, lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lens", default=LENS)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=None, help="default: the deepest fitted layer")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # Token pieces carry bytes no console codepage covers; a decode of real vocabulary has to
    # survive being printed.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    lens = load_lens(a.lens)
    layer = a.layer if a.layer is not None else max(lens.source_layers)
    J = lens.jacobian(layer)
    d = J.shape[0]

    dec = decompose(J)
    M = dec.residual
    K_J = transport_spectrum(J, far=a.far).K
    K_M = transport_spectrum(M, far=a.far).K

    print(f"{a.model}  layer {layer} of {max(lens.source_layers)}   d={d}   "
          f"n_prompts={lens.n_prompts}")
    print(f"identity share {dec.removed_energy:.3f}   alpha {dec.alpha:.4f}")
    print(f"K(J) = {K_J}    K(J - alpha I) = {K_M}    "
          f"modes the identity hides: {K_M - K_J}\n")

    U, S, _ = np.linalg.svd(M, full_matrices=False)
    tok, W_U, g = readout(a.model)

    # Three regimes: resolved on the raw transport, revealed by removing the identity, and past
    # the identity-free floor. The last is the control -- if those looked as coherent as the
    # middle group, the floor would not be marking anything.
    picks = ([("resolved on J", k) for k in range(min(3, K_J))]
             + [("hidden by the identity", k) for k in
                np.linspace(K_J, K_M - 1, min(8, max(0, K_M - K_J)), dtype=int)]
             + [("past the floor", k) for k in (K_M + 10, K_M + 60, d - 1) if k < d])

    rows = []
    for label, k in picks:
        hi, lo = name_mode(U[:, k], W_U, g, tok, a.top)
        rows.append({"regime": label, "mode": int(k), "sigma": float(S[k]),
                     "promotes": hi, "suppresses": lo})
        print(f"[{label}]  mode {k:>4}   sigma {S[k]:.4f}")
        print(f"    promotes  {' | '.join(repr(t) for t in hi)}")
        print(f"    suppresses{' | '.join(repr(t) for t in lo)}\n")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"model": a.model, "layer": layer, "d": d, "alpha": dec.alpha,
             "identity_share": dec.removed_energy, "K_J": K_J, "K_M": K_M, "modes": rows},
            indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
