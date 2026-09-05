"""B4' -- the resolution limit of coverage as a transport-quality read.

B4 ranked two published fits of one transport and asked whether coverage puts the better one
first. It assumed "more fit data is better" as ground truth. Measurement says otherwise
(`exp12_fit_quality.py`): the two fits are separated by ~0.1% of held-out transport error and the
smaller one wins on most layers, so B4 was ranking a coin flip. A binary verdict on an
unresolvable pair says nothing about the instrument.

The replacement asks the question an instrument should be asked: **how large must a difference be
before the read resolves it?**

    ground truth   held-out transport error, || F - H J^T ||_F / || F ||_F, measured -- not
                   assumed. A degraded transport that does not actually measure worse is dropped
                   rather than counted.
    degradations   two, probing opposite failure modes.
                     noise     J + s*G, isotropic -- adds directions that carry nothing
                     rank      best rank-k approximation of J -- removes directions that carry
                   Sweeping the strength sweeps the size of the true gap.
    metric         at each true gap, the fraction of layers where the read orders the pair
                   correctly. Chance is 0.5. The resolution limit is the smallest gap held
                   above chance.
    baselines      coverage against three cheaper reads of the same transported frame, because a
                   sensitivity curve alone does not show that coverage is the thing supplying it.
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
    """Two-sided exact sign test: k successes in n trials against p=1/2."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def degrade(J, kind: str, strength: float, rng):
    """A worse transport, by one of two opposite mechanisms."""
    if kind == "noise":
        G = rng.standard_normal(J.shape)
        return J + strength * (np.linalg.norm(J) / np.linalg.norm(G)) * G
    if kind == "rank":
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        k = max(1, int(round((1.0 - strength) * len(s))))
        return (U[:, :k] * s[:k]) @ Vt[:k]
    raise ValueError(f"unknown degradation {kind!r}")


def transport_error(S, l, J):
    """Held-out relative error carrying layer `l` to the model's own final residual."""
    return float(np.mean([np.linalg.norm(s[-1] - s[l + 1] @ J.T) / np.linalg.norm(s[-1])
                          for s in S]))


def reads(S, l, J):
    """Candidate scores for "is this transport good", all unlabelled, all on one frame."""
    cov, pr, nrm = [], [], []
    for s in S:
        H = s[l + 1]
        P = H @ J.T
        cov.append(je.coverage(H, P).excess)
        sp = je.energy_spectrum(P)
        pr.append(je.participation_ratio(sp))
        nrm.append(float(np.linalg.norm(P) / np.linalg.norm(H)))
    return {"coverage": float(np.mean(cov)),
            "participation": float(np.mean(pr)),
            "norm_ratio": float(np.mean(nrm))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"))
    ap.add_argument("--streams", type=Path, default=Path("streams/qwen35_4b_streams.npz"))
    ap.add_argument("--every", type=int, default=4, help="layer stride, to bound runtime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/ranking_sensitivity.json"))
    a = ap.parse_args(argv)

    lens = je.load_lens(a.lens)
    z = np.load(a.streams)
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    layers = list(lens.source_layers)[::a.every]
    rng = np.random.default_rng(a.seed)
    print(f"{len(S)} held-out streams, {len(layers)} layers "
          f"(stride {a.every}), d={S[0].shape[-1]}")

    grid = [("noise", s) for s in (0.02, 0.05, 0.10, 0.20, 0.40)] + \
           [("rank", s) for s in (0.05, 0.10, 0.25, 0.50, 0.75)]
    keys = ("coverage", "participation", "norm_ratio")

    base = {}
    for l in layers:
        J = lens.jacobian(l)
        base[l] = (J, transport_error(S, l, J), reads(S, l, J))
    print("reference transport measured")

    rows = []
    print()
    print(f"{'degradation':>14}{'true gap':>10}{'coverage':>10}{'particip.':>11}"
          f"{'norm':>8}{'n':>4}{'p(cov)':>9}")
    for kind, st in grid:
        gaps, hit = [], {k: 0 for k in keys}
        n = 0
        for l in layers:
            J, e0, r0 = base[l]
            Jd = degrade(J, kind, st, rng)
            e1 = transport_error(S, l, Jd)
            if e1 <= e0:                     # degradation did not degrade: no ground truth here
                continue
            r1 = reads(S, l, Jd)
            n += 1
            gaps.append((e1 - e0) / e0)
            for k in keys:
                hit[k] += int(r0[k] > r1[k])          # reference should score higher
        if n == 0:
            continue
        g = float(np.mean(gaps))
        p = sign_p(hit["coverage"], n)
        rows.append({"kind": kind, "strength": st, "true_gap": g, "n_layers": n,
                     "acc": {k: hit[k] / n for k in keys}, "p_coverage": p})
        print(f"{kind + ' ' + format(st, '.2f'):>14}{g:>10.4f}"
              f"{hit['coverage']/n:>10.2f}{hit['participation']/n:>11.2f}"
              f"{hit['norm_ratio']/n:>8.2f}{n:>4}{p:>9.4f}" + ("  *" if p < 0.05 else ""),
              flush=True)
        je.dump(a.out, {"lens": str(a.lens), "streams": len(S),
                        "layers": layers, "rows": rows}, complete=False)

    je.dump(a.out, {"lens": str(a.lens), "streams": len(S),
                    "layers": layers, "rows": rows}, complete=True)
    print()
    for kind in ("noise", "rank"):
        ok = [r for r in rows if r["kind"] == kind and r["p_coverage"] < 0.05
              and r["acc"]["coverage"] > 0.5]
        if ok:
            r = min(ok, key=lambda r: r["true_gap"])
            print(f"{kind:>6}: coverage resolves a transport-error gap of "
                  f"{r['true_gap']:.2%} (acc {r['acc']['coverage']:.0%}, "
                  f"p={r['p_coverage']:.4f})")
        else:
            print(f"{kind:>6}: coverage resolves no gap in the swept range above chance")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
