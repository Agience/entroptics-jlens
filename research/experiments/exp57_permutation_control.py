"""The permutation control of section 3.3, both tables, over a range of shuffle seeds.

An entry shuffle is a random draw, so a count read on one is a draw too: one seed's number is not
a property of the transport, and the table reports the range over seeds rather than a single
value. Everything that is not a draw -- K(J), K(J - alpha I), the excess kurtosis and the
participation ratio of the residual -- is read once and is the same on every seed.

    python research/experiments/exp57_permutation_control.py
    python research/experiments/exp57_permutation_control.py --seeds 20

The first table (gpt2 layers 0, 5, 10) is exp1's own read: ``exp1_transport_spectrum.read_layer``
with its control seed, so a row here is the row exp1 writes for that seed. The second table reads
both sides of the comparison on the identity-removed transport: ``J - alpha I`` with
``alpha = tr(J)/d`` (``decompose``), and an entry shuffle of that same residual. Shuffling ``J``
itself would move the diagonal off the diagonal and take the identity away on the control side
only; shuffling the residual removes it from both sides before the comparison is made.

Every count is the ``mp`` read at ``far`` from ``spectrum_under_nulls``, the floor every other
table in the paper uses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp1_transport_spectrum import read_layer                         # noqa: E402

LENS = Path("lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt")


def k_mp(M, far: float, seed: int) -> int:
    return int(je.spectrum_under_nulls(M, far=far, seed=seed)["mp"].K)


def participation(M) -> float:
    s2 = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 * s2).sum())


def span(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", type=Path, default=LENS)
    ap.add_argument("--tails", default="0,5,10", help="layers of the first table")
    ap.add_argument("--inversion", default="7,8,9,10", help="layers of the second table")
    ap.add_argument("--seeds", type=int, default=10, help="shuffle seeds 0..seeds-1")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="the null's own seed, as in exp1")
    ap.add_argument("--out", type=Path, default=Path("results/permutation_control.json"))
    a = ap.parse_args(argv)

    lens = je.load_lens(a.lens)
    seeds = list(range(a.seeds))
    tails = [int(x) for x in a.tails.split(",")]
    inversion = [int(x) for x in a.inversion.split(",")]

    print(f"{a.lens}  far={a.far}  shuffle seeds 0..{a.seeds - 1}\n")
    print(f"{'layer':>6}{'kurtosis':>12}{'K real':>8}{'K shuffled':>12}{'K Gaussian':>12}")
    first = []
    for layer in tails:
        rows = [read_layer(lens, layer, a.far, a.seed, s) for s in seeds]
        shuf = [r["controls"]["shuffled"]["K_mp"] for r in rows]
        gauss = [r["controls"]["gaussian"]["K_mp"] for r in rows]
        first.append({"layer": layer, "excess_kurtosis": rows[0]["excess_kurtosis"],
                      "K": rows[0]["K_mp"], "K_shuffled": shuf, "K_gaussian": gauss})
        print(f"{layer:>6}{rows[0]['excess_kurtosis']:>12.4g}{rows[0]['K_mp']:>8}"
              f"{span(shuf):>12}{span(gauss):>12}", flush=True)

    print(f"\n{'layer':>6}{'K(J)':>7}{'K(shuf J)':>11}{'K(J-aI)':>9}{'K(shuf J-aI)':>14}{'PR(J-aI)':>10}")
    second = []
    for layer in inversion:
        J = np.asarray(lens.jacobian(layer), dtype=np.float64)
        R = je.decompose(J, kind="identity").residual
        shuf, shuf_r = [], []
        for s in seeds:
            # exp1's draw order, so K(shuf J) here is exp1's shuffled control for this seed.
            rng = np.random.default_rng(s)
            je.gaussian_null(J.shape, rng, sigma=je.frobenius_sigma(J))
            S = je.shuffled_entries(J, rng)
            shuf.append(k_mp(S, a.far, a.seed))
            shuf_r.append(k_mp(je.shuffled_entries(R, rng), a.far, a.seed))
        row = {"layer": layer, "K": k_mp(J, a.far, a.seed), "K_shuffled": shuf,
               "K_residual": k_mp(R, a.far, a.seed), "K_shuffled_residual": shuf_r,
               "PR_residual": participation(R)}
        second.append(row)
        print(f"{layer:>6}{row['K']:>7}{span(shuf):>11}{row['K_residual']:>9}"
              f"{span(shuf_r):>14}{row['PR_residual']:>10.3g}", flush=True)

    je.dump(a.out, {"lens": str(a.lens).replace("\\", "/"), "far": a.far, "seed": a.seed,
                    "shuffle_seeds": seeds, "tails": first, "inversion": second}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
