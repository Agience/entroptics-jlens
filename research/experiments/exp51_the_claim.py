"""THE CLAIM: the identity in a lens sets the floor a spectral read of it is counted against.

One command, every lens you have, the whole result. This is the script behind the table the
README leads with.

    python experiments/exp51_the_claim.py
    python experiments/exp51_the_claim.py --lenses "lenses/**/*.pt"

For each lens it reads the deepest fitted layer and reports, side by side, the rank the transport
resolves as it stands and the rank it resolves once the identity the residual stream puts there is
removed. Measured 2026-09-02 across eleven lens files in five families, widths 512 to 4096:

    qwen3-1.7b     identity 0.798    K(J)  3   K(J-aI)  64    21.3x
    qwen3-4b       identity 0.876    K(J)  7   K(J-aI) 149    21.3x
    gemma-3-4b     identity 0.703    K(J) 16   K(J-aI) 165    10.3x
    qwen3.5-4b     identity 0.790    K(J) 25   K(J-aI) 183     7.3x
    gpt2           identity 0.422    K(J)  6   K(J-aI)  39     6.5x
    qwen3.5-0.8b   identity 0.778    K(J) 12   K(J-aI)  68     5.7x
    gemma-3-1b     identity 0.660    K(J) 17   K(J-aI)  64     3.8x
    llama3.1-8b    identity 0.720    K(J) 74   K(J-aI) 269     3.6x
    pythia-70m     identity 0.458    K(J)  2   K(J-aI)   4     2.0x
    gemma-3-270m   identity 0.107    K(J) 39   K(J-aI)  42     1.1x

**The count moves by more than 1.5x in 9 of 10 models.** That is 10 of 11 lens files, and the two
are different numbers: the catalogue carries two independent fits of qwen3.5-4b, and counting that
model twice inflates the headline. They give 7.28x and 7.32x -- which is worth more as a
replication across fits than as an extra row.

**Spearman(identity share, factor) = +0.773** over the eleven files, +0.745 over the ten models.
The size of the change tracks how much of the transport is skip connection. A structure-free
surrogate reproduces that same association, which places it with the floor rather than with the
transport -- see `exp55_structure_free_control.py`, which measures the part that is the
transport's.

Why the correction is exact and not a fit: `alpha = tr(J)/d` is the orthogonal projection of `J`
onto `span(I)` under the Frobenius inner product -- the unique least-squares coefficient against a
basis element the architecture guarantees is present. Nothing is tuned, nothing is trained, and
there is no threshold in it.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def _midranks(a) -> "np.ndarray":
    """Ranks with ties averaged.

    `argsort(argsort(x))` breaks ties by position in the array, which makes the statistic depend
    on row order: three models here share d_model = 2560, and that estimator reports +0.483 for a
    correlation whose mid-rank value is +0.390.
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


def spearman(x, y) -> float:
    return float(np.corrcoef(_midranks(x), _midranks(y))[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenses", default="lenses/*/jlens/Salesforce-wikitext/*.pt",
                    help="glob for the lens files to read")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/the_claim.json"))
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.lenses))
    if not paths:
        raise FileNotFoundError(
            f"no lens matched {a.lenses!r}. Fetch one first: entroptics-jlens fetch gpt2")

    print(f"{'model':<22}{'d':>6}{'layer':>7}{'identity':>10}{'K(J)':>7}{'K(J-aI)':>9}"
          f"{'understated':>13}")
    rows = []
    for p in paths:
        path = Path(p)
        name = path.parts[1] if len(path.parts) > 1 else path.stem
        # A model with more than one published fit gets one headline row, and it has to be the
        # file the catalogue names for that model. Marking the other one by comparing against
        # CATALOG rather than by filename keeps the bare model name on the canonical fit: for
        # qwen3.5-4b that is the n=1000 file, and labelling by "n1000 in the name" put the
        # n=417 fit's numbers in the headline row instead.
        lens = je.load_lens(path)
        canonical = je.CATALOG.get(name)
        if canonical and path.name != canonical[1]:
            name += f"-n{lens.n_prompts}"
        top = lens.source_layers[-1]
        J = lens.jacobian(top)
        dec = je.decompose(J)
        # Exact SVDs on both sides: the fast route squares the small singular values, and the
        # noise floor is read exactly there.
        k_raw = je.transport_spectrum(J, far=a.far, null="mp",
                                      s=np.linalg.svd(J, compute_uv=False)).K
        k_dec = je.transport_spectrum(dec.residual, far=a.far, null="mp",
                                      s=np.linalg.svd(dec.residual, compute_uv=False)).K
        factor = (k_dec / k_raw) if k_raw else float("inf")
        rows.append({"model": name, "d_model": lens.d_model, "layer": top,
                     "identity_energy": dec.removed_energy, "K_raw": k_raw, "K_decomposed": k_dec,
                     "understatement": factor})
        print(f"{name:<22}{lens.d_model:>6}{top:>7}{dec.removed_energy:>10.3f}"
              f"{k_raw:>7}{k_dec:>9}{factor:>12.1f}x", flush=True)
        je.dump(a.out, {"far": a.far, "lenses": rows}, complete=False)

    finite = [r for r in rows if np.isfinite(r["understatement"])]
    ident = [r["identity_energy"] for r in finite]
    fact = [r["understatement"] for r in finite]
    rho = spearman(ident, fact) if len(finite) > 2 else float("nan")
    over = sum(1 for r in finite if r["understatement"] > 1.5)

    # Files and MODELS are counted separately, because they differ: the catalogue carries two
    # independent fits of qwen3.5-4b, and counting that model twice inflates the headline. The
    # figure quoted anywhere should be the model count; the duplicate is a replication, not a row.
    models: dict[str, list[float]] = {}
    for r in finite:
        models.setdefault(re.sub(r"-n\d+$", "", r["model"]), []).append(r["understatement"])
    over_models = sum(1 for v in models.values() if max(v) > 1.5)

    print(f"\nthe count moves by more than 1.5x in {over_models} of {len(models)} models "
          f"({over} of {len(finite)} lens files)")
    for name, vals in models.items():
        if len(vals) > 1:
            print(f"  {name}: {len(vals)} independent fits giving "
                  f"{', '.join(f'{v:.2f}x' for v in vals)} -- it replicates across fits too")
    print(f"identity share at the deepest layer: {min(ident):.3f} to {max(ident):.3f}")
    print(f"Spearman(identity share, factor) = {rho:+.3f}")
    print("\nThe size of the change tracks how much of the transport is skip connection. A")
    print("structure-free surrogate reproduces the same association, so run exp55 to separate")
    print("the floor's response from the transport's. The operation itself is exact and closed")
    print("form: je.decompose(J).residual")

    je.dump(a.out, {"far": a.far, "lenses": rows, "spearman_identity_vs_understatement": rho,
                    "understated_over_1_5x_files": [over, len(finite)],
                    "understated_over_1_5x_models": [over_models, len(models)]}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
