"""The ground truth was missing a term: a Jacobian linearisation is affine, not linear.

`exp16` found the best scalar gain for `H J^T` against the final residual is 2.0 to 4.9, never
near 1. That is not a property of the lens; it is a defect in how it was being scored. A
first-order expansion is

    h_final  ~=  h_final(h0)  +  J (h_l - h0)

so the prediction carries an OFFSET. Scoring `|| F - H J^T ||` omits it, and the fitted scalar
gain has been absorbing the missing intercept -- which is why it sits so far from 1.

The comparison a linearisation deserves removes the offset from both sides, which is centring over
tokens, and only then measures direction and magnitude:

    F_c = F - mean(F),   P_c = P - mean(P),   err = || F_c - c P_c || / || F_c ||

`exp12` used the uncentred form, so its ordering of the two fits rests on a misspecified model.
This recomputes it, and re-asks whether coverage agrees. Coverage is built from a frame's own
resolved basis and is unaffected by this -- so if the corrected error now agrees with coverage,
the disagreement was never in the instrument.
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def sign_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def score(S, l, J):
    """Per-stream affine transport error: offset removed, then best scalar gain."""
    out = []
    for s in S:
        H, F = s[l + 1], s[-1]
        P = H @ J.T
        Fc = F - F.mean(0, keepdims=True)
        Pc = P - P.mean(0, keepdims=True)
        c = float((Fc * Pc).sum() / max((Pc * Pc).sum(), 1e-300))
        out.append(float(np.linalg.norm(Fc - c * Pc) / np.linalg.norm(Fc)))
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/affine.json"))
    a = ap.parse_args(argv)

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    cov = {r["layer"]: r["d_cov"]
           for r in je.load_complete("results/delta_direction.json")["rows"]}
    raw = {r["layer"]: (r["raw_a"] < r["raw_b"])
           for r in je.load_complete("results/gain.json")["rows"]}
    layers = [l for l in A.source_layers if l in cov]

    rows = []
    print(f"{'layer':>6}{'affine n1000':>14}{'affine n417':>13}{'wins':>7}{'p':>8}"
          f"{'affine':>8}{'raw':>6}{'cov':>6}")
    for l in layers:
        ea, eb = score(S, l, A.jacobian(l)), score(S, l, B.jacobian(l))
        w = int((ea < eb).sum())
        aff_big = float(ea.mean()) < float(eb.mean())
        rows.append({"layer": l, "aff_a": float(ea.mean()), "aff_b": float(eb.mean()),
                     "wins": w, "streams": len(S), "p": sign_p(w, len(S)),
                     "aff_prefers_big": aff_big, "raw_prefers_big": raw[l],
                     "cov_prefers_big": cov[l] > 0})
        r = rows[-1]
        print(f"{l:>6}{r['aff_a']:>14.5f}{r['aff_b']:>13.5f}{f'{w}/{len(S)}':>7}"
              f"{r['p']:>8.4f}{'n1000' if aff_big else 'n417':>8}"
              f"{'A' if raw[l] else 'B':>6}{'A' if cov[l] > 0 else 'B':>6}", flush=True)
        je.dump(a.out, {"rows": rows}, complete=False)

    je.dump(a.out, {"rows": rows}, complete=True)
    n = len(rows)
    print()
    print(f"affine error prefers n=1000 on {sum(r['aff_prefers_big'] for r in rows)}/{n}")
    print(f"raw    error prefers n=1000 on {sum(r['raw_prefers_big'] for r in rows)}/{n}")
    print(f"coverage     prefers n=1000 on {sum(r['cov_prefers_big'] for r in rows)}/{n}")
    print(f"affine flips the raw verdict on "
          f"{sum(r['aff_prefers_big'] != r['raw_prefers_big'] for r in rows)}/{n} layers")
    agree = sum(r['aff_prefers_big'] == r['cov_prefers_big'] for r in rows)
    print(f"coverage agrees with the affine error on {agree}/{n} "
          f"(with the raw error: {sum(r['raw_prefers_big'] == r['cov_prefers_big'] for r in rows)}"
          f"/{n})")
    print(f"sign-test p<0.05 on {sum(r['p'] < 0.05 for r in rows)}/{n} layers")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
