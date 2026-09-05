"""Why fit sample size inverts the coverage read: two objectives, not one error.

`exp12` found coverage and held-out transport error ordering the catalogue's two Qwen3.5-4B fits
oppositely, both with complete consistency. Neither is noisy, so one of them is tracking something
other than what "better fit" was taken to mean.

There are two defensible meanings, and they are not the same object:

    estimation   the lens estimates E[dh_final/dh_l], a MEAN JACOBIAN. A better estimate is
                 closer to that mean -- which is what more fit data buys.
    prediction   transport error scores the matrix as a LINEAR PREDICTOR of h_final. Because the
                 layer map is nonlinear, the best linear predictor is a different matrix from the
                 mean Jacobian, and more averaging need not move toward it.

Testable with no autograd and no model. Averaging two estimates of one quantity reduces estimation
variance, so every interior point of

    J(lambda) = lambda * J_n1000 + (1 - lambda) * J_n417

is a lower-variance estimate of the mean Jacobian than either endpoint. Sweep lambda and read both
curves. If coverage peaks in the interior while transport error runs to an endpoint, coverage is
tracking estimation variance and transport error is tracking predictive fit -- and the inversion
is two objectives disagreeing, not a defect in either read.

A shared optimum would refute this and send the question back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/blend.json"))
    a = ap.parse_args(argv)

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    layers = list(A.source_layers)[::a.every]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"{len(S)} streams, {len(layers)} layers; lambda=1 is the n=1000 fit")

    rows = []
    print()
    print("               " + "".join(f"{l:>9.2f}" for l in lams))
    for l in layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        cov, err = [], []
        for lam in lams:
            J = lam * Ja + (1.0 - lam) * Jb
            cov.append(float(np.mean([je.coverage(s[l + 1], s[l + 1] @ J.T).excess for s in S])))
            err.append(float(np.mean([np.linalg.norm(s[-1] - s[l + 1] @ J.T)
                                      / np.linalg.norm(s[-1]) for s in S])))
        rows.append({"layer": l, "lambdas": lams, "coverage": cov, "error": err,
                     "argmax_cov": lams[int(np.argmax(cov))],
                     "argmin_err": lams[int(np.argmin(err))]})
        print(f"L{l:<3} coverage " + "".join(f"{c:>9.4f}" for c in cov)
              + f"   peak {rows[-1]['argmax_cov']:.2f}")
        print("     error    " + "".join(f"{e:>9.5f}" for e in err)
              + f"   min  {rows[-1]['argmin_err']:.2f}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"layers": layers, "rows": rows}, indent=2))

    n = len(rows)
    cov_int = sum(1 for r in rows if 0.0 < r["argmax_cov"] < 1.0)
    err_int = sum(1 for r in rows if 0.0 < r["argmin_err"] < 1.0)
    print()
    print(f"coverage peaks in the interior on {cov_int}/{n} layers "
          f"(endpoints: {sum(1 for r in rows if r['argmax_cov'] == 1.0)} at n=1000, "
          f"{sum(1 for r in rows if r['argmax_cov'] == 0.0)} at n=417)")
    print(f"error  minimises in the interior on {err_int}/{n} layers "
          f"(endpoints: {sum(1 for r in rows if r['argmin_err'] == 1.0)} at n=1000, "
          f"{sum(1 for r in rows if r['argmin_err'] == 0.0)} at n=417)")
    same = sum(1 for r in rows if r["argmax_cov"] == r["argmin_err"])
    print(f"the two objectives share an optimum on {same}/{n} layers")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
