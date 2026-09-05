"""Is B4's ground truth actually true at every layer?

B4 ranks two fits of one transport by coverage and calls the n=1000 fit "better" everywhere,
because more data fits a mean better. Five layers rank the other way, and three mechanisms for
that are ruled out. The remaining possibility is the one
never tested: that at those layers the smaller fit really IS better, and coverage is right.

Directly checkable. ``J_l`` estimates ``E[dh_final/dh_l]``, so the fit that predicts ``h_final``
better from ``h_l`` is the better fit -- measured on held-out streams, against the model's own
final residual, with no coverage anywhere in the loop.

    relative error   || h_final - h_l J_l^T ||_F / || h_final ||_F

If the n=417 fit wins exactly at the reversal layers, the reversals are not a coverage failure
and the guard then on ``Coverage`` was protecting a region that was not the problem.
(Outcome: it was not. See exp17 -- and note this script's own ground truth is
superseded, since it scores an uncentred product against a post-norm target.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def rel_err(H, F, J):
    """How well ``J`` carries ``H`` to ``F``, relative to the size of ``F``."""
    P = H @ J.T
    return float(np.linalg.norm(F - P) / np.linalg.norm(F))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cov", type=Path,
                    default=Path("results/reversal_mechanism.json"))
    ap.add_argument("--out", type=Path, default=Path("results/fit_quality.json"))
    a = ap.parse_args(argv)

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    cov = je.load_complete(a.cov)["rows"]
    d_cov = {r["layer"]: r["d_floor"] for r in cov}
    reversals = {r["layer"] for r in cov if r["d_floor"] < 0}

    rows = []
    print("transport error to the model's own final residual, held-out streams")
    print(f"{'layer':>6}{'err n=1000':>12}{'err n=417':>11}{'winner':>9}"
          f"{'cov says':>10}{'agree':>7}")
    for l in A.source_layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        ea = float(np.mean([rel_err(s[l + 1], s[-1], Ja) for s in S]))
        eb = float(np.mean([rel_err(s[l + 1], s[-1], Jb) for s in S]))
        truth_a = ea < eb                       # does the big fit actually transport better?
        cov_a = d_cov[l] > 0                    # does coverage say the big fit is better?
        rows.append({"layer": l, "err_n1000": ea, "err_n417": eb,
                     "truth_prefers_big": truth_a, "coverage_prefers_big": cov_a,
                     "agree": truth_a == cov_a, "reversal": l in reversals})
        print(f"{l:>6}{ea:>12.5f}{eb:>11.5f}{'n=1000' if truth_a else 'n=417':>9}"
              f"{'n=1000' if cov_a else 'n=417':>10}{'yes' if truth_a == cov_a else 'NO':>7}",
              flush=True)

    n = len(rows)
    big_wins = sum(1 for r in rows if r["truth_prefers_big"])
    agree = sum(1 for r in rows if r["agree"])
    rev = [r for r in rows if r["reversal"]]
    rev_big = sum(1 for r in rev if r["truth_prefers_big"])
    print()
    print(f"assumed ground truth (n=1000 better everywhere) holds on {big_wins}/{n} layers")
    print(f"coverage agrees with measured transport error on {agree}/{n} layers")
    print(f"at the {len(rev)} coverage reversals, the big fit truly wins {rev_big}/{len(rev)}")
    if rev and rev_big < len(rev):
        print(f"=> {len(rev) - rev_big} of the reversals are coverage being RIGHT and the "
              f"assumed ground truth being wrong")
    je.dump(a.out, {"layers": rows, "truth_holds": big_wins,
                    "coverage_agrees": agree, "n": n, "reversals": len(rev),
                    "reversals_truth_big": rev_big}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults,
            ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
