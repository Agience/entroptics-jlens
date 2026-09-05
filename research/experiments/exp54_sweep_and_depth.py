"""The two robustness tables of section 1: the false-alarm sweep and the depth profile.

Both were quoted in the paper with no script behind them, which made the paper's two robustness
arguments the only claims in it a reader could not reproduce. This produces them.

    python research/experiments/exp54_sweep_and_depth.py                  # both tables
    python research/experiments/exp54_sweep_and_depth.py --deepest-only   # the sweep alone

**The sweep is cheap and the depth profile is not.** Changing the false-alarm rate moves only the
Tracy-Widom quantile, not the spectrum, so one pair of SVDs per lens serves all five rates. The
depth profile needs a pair per layer -- 269 layers across the catalogue, and a 4096-wide transport
takes minutes -- so `--max-width` bounds what it will attempt and the output states which lenses
were read. A table that silently covered a subset is what made the original unreproducible.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

RATES = (0.5, 0.05, 0.005, 0.0005, 1e-6)


def counts(J, rates):
    """K(J) and K(J - alpha I) at every rate, from one pair of spectra.

    The floor depends on the rate only through the Tracy-Widom quantile, so the spectra are
    computed once and re-counted; recomputing them per rate would multiply the cost by five and
    return the same singular values.
    """
    M = je.decompose(J).residual
    sJ = np.linalg.svd(J, compute_uv=False)
    sM = np.linalg.svd(M, compute_uv=False)
    return {r: (je.transport_spectrum(J, far=r, null="mp", s=sJ).K,
                je.transport_spectrum(M, far=r, null="mp", s=sM).K) for r in rates}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenses", default="lenses/*/jlens/Salesforce-wikitext/*.pt")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="skip the depth profile for transports wider than this")
    ap.add_argument("--deepest-only", action="store_true", help="the sweep, not the depth profile")
    ap.add_argument("--out", type=Path, default=Path("results/sweep_and_depth.json"))
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.lenses))
    if not paths:
        raise FileNotFoundError(f"no lens matched {a.lenses!r}. Fetch one: entroptics-jlens fetch gpt2")

    print("=== the false-alarm sweep, deepest fitted layer of every lens ===")
    per_rate: dict[float, list[float]] = {r: [] for r in RATES}
    for p in paths:
        lens = je.load_lens(Path(p))
        top = lens.source_layers[-1]
        c = counts(lens.jacobian(top), RATES)
        for r, (kj, km) in c.items():
            per_rate[r].append(km / kj if kj else float("inf"))
        print(f"  {Path(p).parts[1]:<22} " +
              "  ".join(f"{r:g}:{c[r][0]}->{c[r][1]}" for r in RATES), flush=True)

    print(f"\n{'far':>10}{'over 1.5x':>12}{'range':>18}")
    sweep = {}
    for r in RATES:
        v = [x for x in per_rate[r] if np.isfinite(x)]
        over_n = sum(1 for x in v if x > 1.5)
        sweep[r] = {"over": over_n, "n": len(v), "lo": min(v), "hi": max(v)}
        frac = f"{over_n}/{len(v)}"
        span = f"{min(v):.1f}x - {max(v):.1f}x"
        print(f"{r:>10g}{frac:>12}{span:>18}")

    depth = []
    if not a.deepest_only:
        print(f"\n=== the depth profile (transports up to {a.max_width} wide) ===")
        read, skipped = [], []
        for p in paths:
            lens = je.load_lens(Path(p))
            if lens.d_model > a.max_width:
                skipped.append(f"{Path(p).parts[1]}(d={lens.d_model})")
                continue
            read.append(f"{Path(p).parts[1]}({len(lens.source_layers)} layers)")
            last = lens.source_layers[-1]
            for lay in lens.source_layers:
                J = lens.jacobian(lay)
                dec = je.decompose(J)
                kj, km = counts(J, (0.05,))[0.05]
                depth.append({"model": Path(p).parts[1], "layer": lay,
                              "rel": lay / last if last else 0.0,
                              "identity": dec.removed_energy, "K_raw": kj, "K_dec": km,
                              "gain": km / kj if kj else float("inf")})
            print(f"  {Path(p).parts[1]}: {len(lens.source_layers)} layers", flush=True)
        print(f"\nread: {', '.join(read)}")
        print(f"skipped as too wide: {', '.join(skipped) or 'none'}")

        print(f"\n{'relative depth':>16}{'median identity':>18}{'median gain':>14}{'over 1.5x':>12}")
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for lo, hi in bins:
            grp = [r for r in depth if lo <= r["rel"] < hi]
            if not grp:
                continue
            g = [r["gain"] for r in grp if np.isfinite(r["gain"])]
            print(f"{f'{lo:.1f}-{min(hi,1.0):.1f}':>16}"
                  f"{np.median([r['identity'] for r in grp]):>18.3f}"
                  f"{np.median(g):>13.1f}x"
                  f"{f'{sum(1 for x in g if x > 1.5)} / {len(grp)}':>12}")

        over = [r for r in depth if np.isfinite(r["gain"]) and r["gain"] > 1.5]
        under = [r for r in depth if np.isfinite(r["gain"]) and r["gain"] <= 1.5]
        if over and under:
            print(f"\nidentity share where the gain exceeds 1.5x: "
                  f"{min(r['identity'] for r in over):.3f} to {max(r['identity'] for r in over):.3f}")
            print(f"identity share where it does not:            "
                  f"{min(r['identity'] for r in under):.3f} to "
                  f"{max(r['identity'] for r in under):.3f}")

    je.dump(a.out, {"rates": {str(k): v for k, v in sweep.items()},
                    "max_width": a.max_width, "depth": depth}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
