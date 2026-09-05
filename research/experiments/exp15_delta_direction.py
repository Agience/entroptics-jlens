"""What is coverage actually responding to when it prefers the larger fit?

`exp14` ruled out estimation variance: blending the two fits gives a strictly lower-variance
estimate of the mean Jacobian, and coverage is monotone across the blend rather than peaking in
the interior. So coverage is responding to the DIRECTION separating the two fits,

    dJ = J_n1000 - J_n417

not to how good either is as an estimate. This asks what that direction is made of.

The leading candidate is the identity. sec 3 of the paper: J is dominated by a scaled identity,
`alpha = tr(J)/d`, and the exact Frobenius projection onto span(I) splits `J = alpha I + M`. A map
closer to `alpha I` preserves the signal's own subspace almost by definition, so it covers well for
a reason that has nothing to do with transporting anything. If `alpha` is systematically larger in
the fit coverage prefers, the read is partly an identity meter.

Measured per layer:

    d_alpha      alpha(J_n1000) - alpha(J_n417)
    d_cov        mean coverage excess, n=1000 minus n=417   (which fit coverage prefers)
    d_err        mean transport error, n=1000 minus n=417   (negative = n=1000 predicts better)

and the Spearman correlations between them. If `d_alpha` tracks `d_cov` and not `d_err`, the
mechanism is named. A further split isolates it directly: recompute coverage on the identity-free
residual `M` of each fit, which removes the shared identity from both sides. If the preference
survives on `M`, the identity was not the cause.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def _midranks(a) -> "np.ndarray":
    """Ranks with ties averaged.

    `argsort(argsort(x))` breaks a tie by position in the array, so the statistic depends on row
    order. Three of the nine models profiled here share d_model = 2560, where the tie-unaware
    estimator reports +0.483 against a mid-rank +0.390.
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _midranks(x), _midranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results/delta_direction.json"))
    a = ap.parse_args(argv)

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    layers = list(A.source_layers)[::a.every]
    print(f"{len(S)} streams, {len(layers)} layers (stride {a.every})")

    rows = []
    print()
    print(f"{'layer':>6}{'a(n1000)':>11}{'a(n417)':>10}{'d_alpha':>10}"
          f"{'d_cov':>10}{'d_cov|M':>10}{'d_err':>10}")
    for l in layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        da, db = je.decompose(Ja, kind="identity"), je.decompose(Jb, kind="identity")
        cov_a = float(np.mean([je.coverage(s[l + 1], s[l + 1] @ Ja.T).excess for s in S]))
        cov_b = float(np.mean([je.coverage(s[l + 1], s[l + 1] @ Jb.T).excess for s in S]))
        mov_a = float(np.mean([je.coverage(s[l + 1], s[l + 1] @ da.residual.T).excess
                               for s in S]))
        mov_b = float(np.mean([je.coverage(s[l + 1], s[l + 1] @ db.residual.T).excess
                               for s in S]))
        err_a = float(np.mean([np.linalg.norm(s[-1] - s[l + 1] @ Ja.T)
                               / np.linalg.norm(s[-1]) for s in S]))
        err_b = float(np.mean([np.linalg.norm(s[-1] - s[l + 1] @ Jb.T)
                               / np.linalg.norm(s[-1]) for s in S]))
        rows.append({"layer": l, "alpha_a": da.alpha, "alpha_b": db.alpha,
                     "d_alpha": da.alpha - db.alpha, "d_cov": cov_a - cov_b,
                     "d_cov_M": mov_a - mov_b, "d_err": err_a - err_b})
        r = rows[-1]
        print(f"{l:>6}{da.alpha:>11.5f}{db.alpha:>10.5f}{r['d_alpha']:>+10.5f}"
              f"{r['d_cov']:>+10.5f}{r['d_cov_M']:>+10.5f}{r['d_err']:>+10.5f}", flush=True)
        je.dump(a.out, {"layers": layers, "rows": rows}, complete=False)

    je.dump(a.out, {"layers": layers, "rows": rows}, complete=True)
    dal = [r["d_alpha"] for r in rows]
    dcv = [r["d_cov"] for r in rows]
    der = [r["d_err"] for r in rows]
    print()
    print(f"Spearman(d_alpha, d_cov)   {spearman(dal, dcv):>+7.3f}   "
          f"does the identity gap track which fit coverage prefers?")
    print(f"Spearman(d_alpha, d_err)   {spearman(dal, der):>+7.3f}   "
          f"does it track which fit predicts better?")
    print(f"Spearman(d_cov,   d_err)   {spearman(dcv, der):>+7.3f}   "
          f"do the two reads agree at all?")
    print()
    agree_M = sum(1 for r in rows if (r["d_cov_M"] > 0) == (r["d_cov"] > 0))
    print(f"coverage on the identity-free residual M keeps the same preference on "
          f"{agree_M}/{len(rows)} layers")
    print(f"  full J   prefers n=1000 on {sum(1 for r in rows if r['d_cov'] > 0)}/{len(rows)}")
    print(f"  residual prefers n=1000 on {sum(1 for r in rows if r['d_cov_M'] > 0)}/{len(rows)}")
    print(f"  error    prefers n=1000 on {sum(1 for r in rows if r['d_err'] < 0)}/{len(rows)}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
