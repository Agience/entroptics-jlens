"""sec 7.7 was scored against the wrong target. This corrects it.

`hidden_states[-1]` from HuggingFace has the model's final norm **already applied** -- in this
checkpoint its mean per-token norm is 156.1 against 52.1 for its neighbour, a 3x discontinuity
that is exactly the RMSNorm. But `J` maps into the PRE-norm residual: `unembed_fn` applies the
final norm itself, which is what makes composing with it meaningful (paper sec 2.3). So every
transport error in exp12, exp16 and exp17 predicted a normalised target with a map that outputs an
unnormalised one, and the fitted scalar gains of 1.9 to 4.9 were largely that norm ratio.

The pre-norm residual is recoverable without re-running the model. RMSNorm is

    y = (x / rms(x)) * w        elementwise in w

so `y / w` is `x` scaled to unit rms: the DIRECTION of the pre-norm residual, exactly, with only
the per-token scale lost. Direction is also the right target -- `unembed` is linear, so it is
direction that fixes the logits up to a scale.

Scored as cosine between the transported frame and that direction, centred over tokens first,
because both are dominated by a shared component that agrees whatever the transport does. The
centred cosine is the same quantity sec 7.7 reports as explained variance, on the correct target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402
from entroptics_jlens import centred_cosine, final_norm_weight         # noqa: E402

# `final_norm_weight`, `prenorm_direction` and `centred_cosine` live in
# `entroptics_jlens.targets` -- both corrections they encode are traps any
# comparison against the final residual walks into, so they are library code.

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--streams", type=Path, default=Path("streams/qwen35_4b_streams.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/prenorm_target.json"))
    a = ap.parse_args(argv)

    w = final_norm_weight()
    print(f"final norm weight: d={w.size}, min {w.min():.4f}, max {w.max():.4f}, "
          f"|w| near zero: {(np.abs(w) < 1e-3).sum()}")
    if (np.abs(w) < 1e-6).any():
        raise ValueError("final norm weight has zero entries; the direction is not recoverable")

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load(a.streams)
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    # the pre-norm final residual, recovered up to per-token scale
    X = [je.prenorm_direction(s[-1], w) for s in S]
    post = [s[-1] for s in S]

    rows = []
    print()
    print(f"{'layer':>6}{'cos pre-norm':>14}{'cos post-norm':>15}{'n417 pre':>10}"
          f"{'wins':>7}")
    for l in A.source_layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        ca = [centred_cosine(s[l + 1] @ Ja.T, x) for s, x in zip(S, X)]
        cb = [centred_cosine(s[l + 1] @ Jb.T, x) for s, x in zip(S, X)]
        cp = [centred_cosine(s[l + 1] @ Ja.T, y) for s, y in zip(S, post)]
        w417 = int(sum(1 for u, v in zip(ca, cb) if v > u))
        rows.append({"layer": l, "cos_a": float(np.mean(ca)), "cos_b": float(np.mean(cb)),
                     "cos_post_a": float(np.mean(cp)), "n417_wins": w417, "streams": len(S)})
        r = rows[-1]
        print(f"{l:>6}{r['cos_a']:>14.4f}{r['cos_post_a']:>15.4f}"
              f"{r['cos_b']:>10.4f}{f'{w417}/{len(S)}':>7}", flush=True)
        je.dump(a.out, {"rows": rows}, complete=False)

    je.dump(a.out, {"rows": rows}, complete=True)
    n = len(rows)
    print()
    print(f"n=417 is the better linearisation on {sum(1 for r in rows if r['cos_b'] > r['cos_a'])}"
          f"/{n} layers, on the corrected target")
    d = [r["cos_a"] - r["cos_post_a"] for r in rows]
    print(f"correcting the target changes the centred cosine by {np.mean(d):+.4f} on average "
          f"(min {min(d):+.4f}, max {max(d):+.4f})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
