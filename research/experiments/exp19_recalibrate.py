"""B4' recalibrated on the corrected ground truth, and the `resolves_gap` threshold derived.

Two debts from earlier in this session.

`exp13` measured how large a transport-quality gap must be before coverage resolves it, and quoted
the answer in *uncentred transport error against the post-norm residual* — the metric `exp17` and
`exp18` then showed to be misspecified twice over (a missing affine offset, and a target carrying
the model's final norm). The sensitivity curve's shape does not depend on that, but the numbers on
its x-axis do, so the limit is restated here against the corrected ground truth: centred cosine to
the pre-norm residual direction.

And `Coverage.resolves_gap` ships a hardcoded 0.005 justified by hand. A threshold on the read's
own scale should be *calibrated* — read off the relationship between the coverage separation and
whether the ordering is actually right — not asserted.

Both come from one sweep: degrade a known-good transport by a measured amount, record the true gap,
the coverage separation, and whether coverage ordered the pair correctly.
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402
from entroptics_jlens import centred_cosine, final_norm_weight         # noqa: E402


def sign_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/recalibrate.json"))
    a = ap.parse_args(argv)

    w = final_norm_weight()
    lens = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                        "Qwen3.5-4B_jacobian_lens_n1000.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    X = [je.prenorm_direction(s[-1], w) for s in S]
    layers = list(lens.source_layers)[::a.every]
    rng = np.random.default_rng(a.seed)
    noise = (0.02, 0.05, 0.10, 0.20, 0.40)
    rank = (0.05, 0.10, 0.25, 0.50, 0.75)
    print(f"{len(S)} streams, {len(layers)} layers, corrected ground truth "
          f"(centred cosine to the pre-norm direction)")

    def read(l, J):
        """Ground truth, plus all three candidate reads, for one transport at one layer.

        The baselines ride along because a sensitivity curve alone does not show that coverage is
        the thing supplying it, and their accuracy depends on the ground truth -- so they have to
        be rescored whenever it changes.
        """
        cos, exc, pr, nrm = [], [], [], []
        for s, x in zip(S, X):
            H = s[l + 1]
            P = H @ J.T
            cos.append(centred_cosine(P, x))
            exc.append(je.coverage(H, P).excess)
            pr.append(je.participation_ratio(je.energy_spectrum(P)))
            nrm.append(float(np.linalg.norm(P) / np.linalg.norm(H)))
        return (float(np.mean(cos)),
                {"coverage": float(np.mean(exc)), "participation": float(np.mean(pr)),
                 "norm_ratio": float(np.mean(nrm))})

    pairs = []
    for l in layers:
        J = lens.jacobian(l)
        cos0, r0 = read(l, J)
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)      # one SVD, reused by every rank cut
        cand = [("noise", t, J + t * (np.linalg.norm(J) / np.linalg.norm(g)) * g)
                for t, g in ((t, rng.standard_normal(J.shape)) for t in noise)]
        cand += [("rank", t, (U[:, :k] * sv[:k]) @ Vt[:k])
                 for t, k in ((t, max(1, int(round((1.0 - t) * len(sv))))) for t in rank)]
        for kind, t, Jd in cand:
            cos1, r1 = read(l, Jd)
            if cos1 >= cos0:                        # degradation did not degrade: no ground truth
                continue
            pairs.append({"layer": l, "kind": kind, "strength": t,
                          "gap": (cos0 - cos1) / cos0, "d_excess": r0["coverage"] - r1["coverage"],
                          "correct": bool(r0["coverage"] > r1["coverage"]),
                          "acc": {k: bool(r0[k] > r1[k]) for k in r0}})
        print(f"  layer {l:>2}: {sum(1 for p in pairs if p['layer'] == l)} usable pairs",
              flush=True)
        je.dump(a.out, {"pairs": pairs}, complete=False)

    print()
    print("sensitivity, on the corrected ground truth")
    print(f"{'degradation':>14}{'true gap':>11}{'coverage':>10}{'particip.':>11}"
          f"{'norm':>8}{'n':>4}{'p':>9}")
    rows = []
    for kind in ("noise", "rank"):
        for t in (noise if kind == "noise" else rank):
            g = [p for p in pairs if p["kind"] == kind and p["strength"] == t]
            if not g:
                continue
            k = sum(p["correct"] for p in g)
            rows.append({"kind": kind, "strength": t,
                         "gap": float(np.mean([p["gap"] for p in g])),
                         "acc": {m: sum(p["acc"][m] for p in g) / len(g)
                                 for m in ("coverage", "participation", "norm_ratio")},
                         "n": len(g), "p": sign_p(k, len(g))})
            r = rows[-1]
            print(f"{kind + ' ' + format(t, '.2f'):>14}{r['gap']:>11.4f}"
                  f"{r['acc']['coverage']:>10.2f}{r['acc']['participation']:>11.2f}"
                  f"{r['acc']['norm_ratio']:>8.2f}{r['n']:>4}{r['p']:>9.4f}"
                  + ("  *" if r["p"] < 0.05 else ""))
    for kind in ("noise", "rank"):
        ok = [r for r in rows if r["kind"] == kind and r["p"] < 0.05
              and r["acc"]["coverage"] > 0.5]
        if ok:
            r = min(ok, key=lambda r: r["gap"])
            print(f"  {kind}: resolves a gap of {r['gap']:.2%} "
                  f"(acc {r['acc']['coverage']:.0%}, p={r['p']:.4f})")

    print()
    print("threshold calibration: accuracy against the coverage separation itself")
    mag = np.array([abs(p["d_excess"]) for p in pairs])
    ok = np.array([p["correct"] for p in pairs])
    edges = [0.0, 0.001, 0.002, 0.005, 0.010, 0.020, np.inf]
    print(f"{'|d_excess|':>18}{'accuracy':>10}{'n':>5}{'p':>9}")
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag < hi)
        if not m.any():
            continue
        k = int(ok[m].sum())
        bins.append({"lo": lo, "hi": None if hi == np.inf else hi, "acc": k / int(m.sum()),
                     "n": int(m.sum()), "p": sign_p(k, int(m.sum()))})
        lab = f"[{lo:.3f}, {'inf' if hi == np.inf else format(hi, '.3f')})"
        print(f"{lab:>18}{bins[-1]['acc']:>10.2f}{bins[-1]['n']:>5}{bins[-1]['p']:>9.4f}"
              + ("  *" if bins[-1]["p"] < 0.05 else ""))
    good = [b for b in bins if b["p"] < 0.05 and b["acc"] > 0.5]
    thr = min((b["lo"] for b in good), default=None)
    print()
    print(f"lowest separation bin that orders reliably: "
          f"{'none in range' if thr is None else format(thr, '.3f')}")
    je.dump(a.out, {"pairs": pairs, "rows": rows, "bins": bins,
                    "threshold": thr}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
