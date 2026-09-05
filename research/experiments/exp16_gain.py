"""Is the disagreement a gain mismatch?

Two candidates are gone: estimation variance (`exp14`, coverage is monotone across the blend) and
the identity component (`exp15`, the preferred fit has the SMALLER alpha and the preference
survives removing alpha I). What remains is structural, and it is the clearest difference between
the two reads:

    coverage         a subspace overlap. Built from orthonormal bases, so it is blind to the
                     singular values entirely -- scaling J by any c > 0 changes it not at all.
    transport error  || F - H J^T ||, dominated by magnitude. A map with the right directions and
                     the wrong gain is scored as badly wrong.

So a systematic gain difference between the fits would push the two reads in opposite directions
while both stayed perfectly consistent -- exactly the observed pattern. `exp15` already shows the
n=1000 fit has a smaller alpha at every layer, which is a gain difference.

Test: give each fit its own best scalar gain before scoring, `c* = <F, HJ^T> / ||HJ^T||^2`, which
is the least-squares optimum and costs nothing. This removes gain from the comparison and leaves
direction. If the transport-error preference flips toward the fit coverage prefers, the two reads
never disagreed about the map -- only about whether gain counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def errors(S, l, J):
    """Relative transport error, raw and at each stream's own best scalar gain."""
    raw, fit, gains = [], [], []
    for s in S:
        H, F = s[l + 1], s[-1]
        P = H @ J.T
        nf = float(np.linalg.norm(F))
        c = float((F * P).sum() / max((P * P).sum(), 1e-300))
        raw.append(float(np.linalg.norm(F - P) / nf))
        fit.append(float(np.linalg.norm(F - c * P) / nf))
        gains.append(c)
    return float(np.mean(raw)), float(np.mean(fit)), float(np.mean(gains))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/gain.json"))
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
    layers = [l for l in A.source_layers[::a.every] if l in cov]
    print(f"{len(S)} streams, {len(layers)} layers")

    rows = []
    print()
    print(f"{'layer':>6}{'raw n1000':>11}{'raw n417':>10}{'gain-fit n1000':>16}"
          f"{'gain-fit n417':>15}{'c n1000':>9}{'c n417':>8}{'flips':>7}")
    for l in layers:
        ra, fa, ca = errors(S, l, A.jacobian(l))
        rb, fb, cb = errors(S, l, B.jacobian(l))
        raw_big = ra < rb                    # raw error prefers n=1000
        fit_big = fa < fb                    # gain-corrected error prefers n=1000
        cov_big = cov[l] > 0                 # coverage prefers n=1000
        rows.append({"layer": l, "raw_a": ra, "raw_b": rb, "fit_a": fa, "fit_b": fb,
                     "c_a": ca, "c_b": cb, "raw_prefers_big": raw_big,
                     "fit_prefers_big": fit_big, "cov_prefers_big": cov_big})
        print(f"{l:>6}{ra:>11.5f}{rb:>10.5f}{fa:>16.5f}{fb:>15.5f}"
              f"{ca:>9.4f}{cb:>8.4f}{'YES' if raw_big != fit_big else '-':>7}", flush=True)
        je.dump(a.out, {"rows": rows}, complete=False)

    je.dump(a.out, {"rows": rows}, complete=True)
    n = len(rows)
    print()
    print(f"raw error           prefers n=1000 on {sum(r['raw_prefers_big'] for r in rows)}/{n}")
    print(f"gain-corrected      prefers n=1000 on {sum(r['fit_prefers_big'] for r in rows)}/{n}")
    print(f"coverage            prefers n=1000 on {sum(r['cov_prefers_big'] for r in rows)}/{n}")
    print(f"coverage agrees with raw error       on "
          f"{sum(r['cov_prefers_big'] == r['raw_prefers_big'] for r in rows)}/{n}")
    print(f"coverage agrees with gain-corrected  on "
          f"{sum(r['cov_prefers_big'] == r['fit_prefers_big'] for r in rows)}/{n}")
    print(f"mean best gain: n=1000 {np.mean([r['c_a'] for r in rows]):.4f}, "
          f"n=417 {np.mean([r['c_b'] for r in rows]):.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
