"""How much of the change in resolved count is a property of the transport?

Removing the identity changes K by 2x to 21x. A derived floor rises with any added identity,
whatever sits underneath it, so part of that range is what the estimator does to any matrix of the
same shape. This measures the part that belongs to the transport, against two surrogates that
preserve different invariants.

    python research/experiments/exp55_structure_free_control.py
    python research/experiments/exp55_structure_free_control.py --seeds 10

**Two surrogates, because the answer depends on which invariants are held.** The `mp` floor's only
free input is a de-biased median row energy, so a surrogate's treatment of that statistic decides
what the comparison measures.

  * `spectrum` -- (U s) V^T with Haar U and V. Preserves the singular values and the Frobenius
    norm; redistributes energy across rows, so the median row energy moves and the floor moves
    with it.
  * `rotation` -- M @ Q with Haar orthogonal Q. Preserves the singular values, every row norm
    exactly, and therefore the floor's variance estimate; destroys the alignment between the
    transport's input and output directions. It keeps the left singular vectors, so it retains the
    massive-activation row structure and bounds the excess from below.

Both add alpha*I back before the raw read, so the identity is present on both sides exactly as it
is in the published file.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def K(A) -> int:
    return je.transport_spectrum(A, null="mp", s=np.linalg.svd(A, compute_uv=False)).K


def surrogates(M, alpha, rng):
    """The two structure-free stand-ins, each with the identity added back."""
    d = M.shape[0]
    s = np.linalg.svd(M, compute_uv=False)
    U = np.linalg.qr(rng.standard_normal((d, d)))[0]
    V = np.linalg.qr(rng.standard_normal((d, d)))[0]
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return {"spectrum": (U * s) @ V.T, "rotation": M @ Q}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenses", default="lenses/*/jlens/Salesforce-wikitext/*.pt")
    ap.add_argument("--seeds", type=int, default=5, help="draws per surrogate")
    ap.add_argument("--out", type=Path, default=Path("results/structure_free_control.json"))
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.lenses))
    if not paths:
        raise FileNotFoundError(
            f"no lens matched {a.lenses!r}. Fetch one first: entroptics-jlens fetch gpt2")

    print(f"{'model':<22}{'share':>7}{'gain':>8}"
          f"{'spectrum':>10}{'excess':>9}{'rotation':>10}{'excess':>9}")
    rows = []
    for p in paths:
        path = Path(p)
        name = path.parts[1] if len(path.parts) > 1 else path.stem
        lens = je.load_lens(path)
        canonical = je.CATALOG.get(name)
        if canonical and path.name != canonical[1]:
            name += f"-n{lens.n_prompts}"
        J = lens.jacobian(lens.source_layers[-1])
        dec = je.decompose(J)
        M, alpha, d = dec.residual, dec.alpha, J.shape[0]
        kj, km = K(J), K(M)
        gain = km / kj if kj else float("inf")

        got: dict[str, list[float]] = {"spectrum": [], "rotation": []}
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            for kind, S in surrogates(M, alpha, rng).items():
                kraw = K(S + alpha * np.eye(d))
                got[kind].append(K(S) / kraw if kraw else float("inf"))
        med = {k: float(np.median(v)) for k, v in got.items()}
        row = {"model": name, "d_model": d, "identity_energy": dec.removed_energy,
               "K_raw": kj, "K_decomposed": km, "gain": gain,
               "surrogate": med, "draws": got,
               "excess": {k: gain / v if v else float("inf") for k, v in med.items()}}
        rows.append(row)
        print(f"{name:<22}{dec.removed_energy:>7.3f}{gain:>7.2f}x"
              f"{med['spectrum']:>9.2f}x{row['excess']['spectrum']:>8.2f}x"
              f"{med['rotation']:>9.2f}x{row['excess']['rotation']:>8.2f}x", flush=True)
        je.dump(a.out, {"seeds": a.seeds, "lenses": rows}, complete=False)

    for kind in ("spectrum", "rotation"):
        v = [r["excess"][kind] for r in rows if np.isfinite(r["excess"][kind])]
        print(f"\nmedian excess over the {kind} surrogate = {np.median(v):.2f}x"
              f"   (range {min(v):.2f}x to {max(v):.2f}x)")

    lo = np.median([r["excess"]["rotation"] for r in rows])
    hi = np.median([r["excess"]["spectrum"] for r in rows])
    print(f"\nThe excess attributable to the transport lies between {min(lo, hi):.2f}x and "
          f"{max(lo, hi):.2f}x,")
    print("the two surrogates differing in whether the floor's own variance input is held fixed.")

    je.dump(a.out, {"seeds": a.seeds, "lenses": rows,
                    "median_excess": {k: float(np.median([r["excess"][k] for r in rows]))
                                      for k in ("spectrum", "rotation")}}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
