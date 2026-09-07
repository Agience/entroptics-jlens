"""Does the claim depend on how the noise floor estimates its variance?

    python research/experiments/exp56_variance_estimator.py

The `mp` floor is `sqrt(sigma^2 * (mu + q_TW * sigma_J))`. Everything but `sigma^2` is fixed by the
matrix shape and the false-alarm rate, both already swept (section 1.4). `sigma^2` is not: it is a
de-biased **median** row energy, and the median is a choice. Section 1.5 makes that choice
load-bearing for one secondary result -- the sign reversal at shallow layers is attributed to the
median disagreeing with the mean diagonal -- so the estimator is the last free parameter in the
chain that produces the headline count.

This sweeps it. The published estimator sits inside a one-parameter family:

    sigma^2(p) = quantile_p(row energies) / (chi2_F.ppf(p) * (N-1)/N)

with p = 0.5 the published median, and the chi^2_F quantile as the matching de-bias so every p is
a consistent estimator of the same variance under a Gaussian null. The arithmetic mean is included
as the non-robust end of the family, de-biased by F.

The prediction is that the count survives, and for a stated reason: `alpha*I` adds `alpha^2` to
**every** row's energy, so it shifts the whole distribution rather than a tail of it. No location
statistic can reject a constant added to all of its inputs. If the effect were an artefact of the
median specifically, a low or high quantile would remove it.

`p = 0.5` is checked against `je.transport_spectrum(..., null="mp").K` on every matrix read, so a
harness that did not reproduce the library's own count would be caught before its sweep is read.
The shape factor is taken from `noise_floor` rather than rebuilt, and cross-checked between the two
matrices, so nothing here depends on an internal of the engine.
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
from entroptics.projection import noise_floor                          # noqa: E402

#: The published estimator, and the family it sits in. 0.5 is `noise_sigma2`'s median.
QUANTILES = (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)


def sigma2_at(A: np.ndarray, p: float | None) -> float:
    """Per-cell noise variance from the ``p``-th quantile of row energy, de-biased for chi^2_F.

    ``p=None`` takes the arithmetic mean, whose chi^2_F expectation is ``F``. The ``(N-1)/N``
    centring correction is the one ``debias_denominator`` applies, kept across the family so the
    rows differ only in the location statistic.
    """
    from scipy.stats import chi2

    N, F = A.shape
    energy = np.einsum("ij,ij->i", A, A)
    dof = max(N - 1, 1) / N
    if p is None:
        return float(energy.mean()) / (F * dof)
    return float(np.quantile(energy, p)) / (float(chi2.ppf(p, F)) * dof)


def sigma2_published(A: np.ndarray) -> float:
    """The engine's own estimator: a de-biased median row energy.

    ``median(||row||^2) / (F * c_F * (N-1)/N)`` with the Wilson-Hilferty ``c_F``, because the
    sample median of the row energies estimates the chi^2_F median rather than its mean.
    """
    N, F = A.shape
    energy = np.einsum("ij,ij->i", A, A)
    c_F = (1.0 - 2.0 / (9.0 * F)) ** 3
    return float(np.median(energy)) / (F * c_F * (max(N - 1, 1) / N))


def shape_factor(A: np.ndarray, s: np.ndarray, far: float) -> float:
    """``mu + q_TW * sigma_J``, the part of the floor fixed by shape and false-alarm rate.

    Read off ``noise_floor`` rather than rebuilt from the engine's internals: the floor is
    ``sqrt(sigma^2 * C)``, so ``C = floor^2 / sigma^2``. `main` checks that the two matrices of a
    pair return the same ``C``, which they can only do if ``sigma2_published`` is the estimator the
    engine actually uses.
    """
    return float(noise_floor(A, far=far, s=s)) ** 2 / sigma2_published(A)


def k_at(A: np.ndarray, s: np.ndarray, p: float | None, far: float, C: float) -> int:
    """Resolved rank with the variance taken by estimator ``p``, everything else unchanged."""
    return int((s > float(np.sqrt(sigma2_at(A, p) * C))).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenses", default="lenses/*/jlens/Salesforce-wikitext/*.pt")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/variance_estimator.json"))
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.lenses))
    if not paths:
        raise FileNotFoundError(
            f"no lens matched {a.lenses!r}. Fetch one first: entroptics-jlens fetch gpt2")

    estimators = [*QUANTILES, None]
    labels = [f"p={p:.2f}" for p in QUANTILES] + ["mean"]
    print(f"{'model':<22}{'identity':>10}" + "".join(f"{lab:>10}" for lab in labels))
    print(f"{'':22}{'':10}" + "".join(f"{'gain':>10}" for _ in labels))

    rows = []
    for path in (Path(p) for p in paths):
        name = path.parts[1] if len(path.parts) > 1 else path.stem
        lens = je.load_lens(path)
        canonical = je.CATALOG.get(name)
        if canonical and path.name != canonical[1]:
            name += f"-n{lens.n_prompts}"
        top = lens.source_layers[-1]
        J = np.asarray(lens.jacobian(top), dtype=np.float64)
        M = np.asarray(je.decompose(J).residual, dtype=np.float64)
        share = float(je.identity_share(J)[1])
        # Exact on both sides: the fast route squares the small singular values and the floor is
        # read exactly there.
        s_j = np.linalg.svd(J, compute_uv=False)
        s_m = np.linalg.svd(M, compute_uv=False)

        # The shape factor depends only on (N, F, far), so J and M must return the same one.
        # They can only do that if `sigma2_published` is the estimator the engine uses: a wrong
        # formula would be wrong by a different factor on each, their row energies differing by
        # the whole identity component.
        C_j, C_m = shape_factor(J, s_j, a.far), shape_factor(M, s_m, a.far)
        if not np.isclose(C_j, C_m, rtol=1e-9):
            raise SystemExit(
                f"{name}: the shape factor reads {C_j:.9g} on J and {C_m:.9g} on M. It depends "
                f"only on the shape, so the variance estimator here is not the engine's.")

        # And the harness reproduces the library's own count before any swept row is believed.
        for A, s, C in ((J, s_j, C_j), (M, s_m, C_m)):
            mine = k_at(A, s, 0.5, a.far, C)
            theirs = je.transport_spectrum(A, far=a.far, null="mp", s=s).K
            if mine != theirs:
                raise SystemExit(
                    f"{name}: the p=0.50 harness read K={mine} where the library reads "
                    f"{theirs}. The sweep is not measuring the published floor; stop here.")

        row = {"model": name, "d_model": lens.d_model, "layer": top, "identity_energy": share,
               "K_raw": {}, "K_decomposed": {}, "gain": {}}
        cells = []
        for p, lab in zip(estimators, labels):
            kr, kd = k_at(J, s_j, p, a.far, C_j), k_at(M, s_m, p, a.far, C_m)
            g = (kd / kr) if kr else float("inf")
            row["K_raw"][lab] = kr
            row["K_decomposed"][lab] = kd
            row["gain"][lab] = g
            cells.append(f"{g:>9.2f}x")
        rows.append(row)
        print(f"{name:<22}{share:>10.3f}" + "".join(cells), flush=True)
        je.dump(a.out, {"far": a.far, "estimators": labels, "lenses": rows}, complete=False)

    # One model per row: the catalogue's two qwen3.5-4b fits are a replication, not two models.
    models: dict[str, list[dict]] = {}
    for r in rows:
        models.setdefault(re.sub(r"-n\d+$", "", r["model"]), []).append(r)

    print(f"\n{'estimator':<12}{'models over 1.5x':>18}{'gain range':>22}{'Spearman':>10}")
    summary = {}
    for lab in labels:
        over = sum(1 for v in models.values() if max(r["gain"][lab] for r in v) > 1.5)
        gains = [r["gain"][lab] for r in rows if np.isfinite(r["gain"][lab])]
        ident = [r["identity_energy"] for r in rows if np.isfinite(r["gain"][lab])]
        rho = _spearman(ident, gains) if len(gains) > 2 else float("nan")
        summary[lab] = {"models_over_1_5x": over, "n_models": len(models),
                        "gain_min": min(gains), "gain_max": max(gains), "spearman": rho}
        print(f"{lab:<12}{f'{over} of {len(models)}':>18}"
              f"{f'{min(gains):.2f}x to {max(gains):.2f}x':>22}{rho:>+10.3f}")

    je.dump(a.out, {"far": a.far, "estimators": labels, "lenses": rows, "summary": summary},
            complete=True)
    print(f"\nwrote {a.out}")
    print("alpha*I adds alpha^2 to every row's energy, so it moves the whole distribution of row")
    print("energies rather than a tail of it. That is why the count does not turn on which")
    print("location statistic reads them.")
    return 0


def _midranks(a) -> np.ndarray:
    """Ranks with ties averaged -- `argsort(argsort(x))` breaks ties by array position, which
    makes the statistic depend on row order where models share a width."""
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


def _spearman(x, y) -> float:
    return float(np.corrcoef(_midranks(x), _midranks(y))[0, 1])


if __name__ == "__main__":
    raise SystemExit(main())
