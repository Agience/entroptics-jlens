"""Experiment 2 -- the depth profile of the residual transport, across models.

Experiment 1 asked how many modes a transport resolves against a noise floor. That question is
answered and the answer is that it is the wrong question: a corpus-averaged Jacobian has had its
noise averaged out (two independent fits of Qwen3.5-4B correlate 0.9999 by layer 30), so there
is no bulk whose edge a floor could find.

What survives is threshold-free, and it needs the architectural component removed first:

  alpha = tr(J)/d          the exact Frobenius projection of J onto span(I). A residual stream
                           adds h_l downstream, so J -> I with depth: 79% of ||J||_F^2 at
                           Qwen3.5-4B layer 30, where the MEDIAN singular value equals alpha.
  M = J - alpha I          the residual transport.
  PR(M)                    participation ratio, (sum s^2)^2 / sum s^4. An energy-weighted
                           effective rank with no threshold in it.
  H2(M)                    Shannon effective rank, 2^H(p) for p_k = s_k^2 / sum s^2.

On Qwen3.5-4B, PR(M) traces an arc peaking at layer 26 of 31 and collapsing at both ends, while
PR(J) undecomposed climbs monotonically and shows no band at all. This experiment asks whether
that arc is a property of one model or of the architecture, by running the same profile across
the published catalogue and plotting against RELATIVE depth so models of different depth are
comparable.

Usage
-----
    python experiments/fetch_lens.py gpt2 pythia-70m gemma-3-270m qwen3.5-0.8b
    python experiments/exp2_depth_profile.py --lenses lenses --live results/depth.html \
        --out results/depth.json

Reads every lens it is pointed at, all layers, no controls needed: nothing here thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
import depth_report                                                    # noqa: E402


from entroptics_jlens import participation_ratio, shannon_rank, energy_spectrum  # noqa: E402


def profile_lens(path: Path, name: str, on_layer=None) -> dict:
    lens = je.load_lens(path)
    n = len(lens.source_layers)
    rows = []
    partial = {"name": name, "d_model": lens.d_model, "n_prompts": lens.n_prompts,
               "n_layers": n, "peak_layer": -1, "peak_depth": 0.0, "peak_PR_M": 0.0,
               "layers": rows}
    for i, l in enumerate(lens.source_layers):
        t0 = time.time()
        J = lens.jacobian(l)
        dec = je.decompose(J, kind="identity")
        # eigvalsh route: 3.2x the SVD at d=2560, and PR/H2 weight by s^2 so they agree to
        # 1e-15. Never used where a noise floor is involved -- see spectra.energy_spectrum.
        sJ = energy_spectrum(J)
        sM = energy_spectrum(dec.residual)
        row = {"layer": l,
               "depth": (l / (lens.source_layers[-1] or 1)),
               "alpha": dec.alpha,
               "identity_energy": dec.removed_energy,
               "PR_J": participation_ratio(sJ),
               "PR_M": participation_ratio(sM),
               "H2_M": shannon_rank(sM),
               "sigma_top_M": float(sM[0]),
               "seconds": time.time() - t0}
        rows.append(row)
        best = max(rows, key=lambda r: r["PR_M"])
        partial["peak_layer"], partial["peak_PR_M"] = best["layer"], best["PR_M"]
        # Peak depth is relative to the layers READ SO FAR, not to the model's depth: a partial
        # peak at 0.00 would otherwise read as the finding it is not (small models genuinely
        # peak at layer 0). Marked partial so the page can say so.
        partial["peak_depth"] = best["layer"] / (rows[-1]["layer"] or 1)
        partial["partial"] = True
        if on_layer:
            on_layer(partial, i + 1, n, row)
    peak = max(rows, key=lambda r: r["PR_M"])
    return {"name": name, "path": str(path), "d_model": lens.d_model,
            "n_prompts": lens.n_prompts, "n_layers": n,
            "peak_layer": peak["layer"], "peak_depth": peak["depth"],
            "peak_PR_M": peak["PR_M"], "layers": rows}


def _midranks(a) -> "np.ndarray":
    """Ranks with ties averaged.

    `argsort(argsort(x))` breaks a tie by position in the array, so the statistic depends on row
    order. Three of the nine models profiled here share d_model = 2560, where the tie-unaware
    estimator reports +0.483 against a mid-rank +0.390.
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
    """Rank correlation. Spearman rather than Pearson because the claim is monotonicity, not
    linearity, and n is small enough that one outlier would dominate a Pearson fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx, ry = _midranks(x), _midranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def trend_summary(profiles: list) -> str:
    """Is the peak's position actually monotone in scale, or does the table just look that way?

    Reported as rank correlations because with fewer than ten models, width, depth and family all
    covary and no single driver can be identified. These numbers describe the sample; they do not
    isolate a cause.
    """
    if len(profiles) < 3:
        return "\ntrend: fewer than 3 models -- nothing to correlate."
    d = [p["d_model"] for p in profiles]
    nl = [p["n_layers"] for p in profiles]
    depth = [p["peak_depth"] for p in profiles]
    height = [p["peak_PR_M"] for p in profiles]
    lines = [f"\ntrend over {len(profiles)} models (Spearman rank correlation):",
             f"  peak relative depth vs d_model    rho = {spearman(d, depth):+.3f}",
             f"  peak relative depth vs n_layers   rho = {spearman(nl, depth):+.3f}",
             f"  peak PR(M)          vs d_model    rho = {spearman(d, height):+.3f}",
             f"  peak PR(M)          vs n_layers   rho = {spearman(nl, height):+.3f}"]
    fams = {}
    for p in profiles:
        fams.setdefault(p["name"].split("-")[0], []).append(p)
    multi = {k: v for k, v in fams.items() if len(v) > 1}
    if multi:
        lines.append("  within family (peak PR(M) by increasing d_model):")
        for k, v in sorted(multi.items()):
            v = sorted(v, key=lambda q: q["d_model"])
            lines.append(f"    {k:<10} " + "  ".join(
                f"{q['name']}={q['peak_PR_M']:.0f}@{q['peak_depth']:.2f}" for q in v))
    lines.append("  NOTE width, depth and family covary across this sample; these correlations")
    lines.append("       describe it and identify no driver.")
    return "\n".join(lines)


def discover(root: Path) -> list[tuple[Path, str]]:
    """Every lens.pt under `root`, named by the catalogue key its path sits in."""
    found = []
    for p in sorted(root.rglob("*_jacobian_lens*.pt")):
        try:
            key = p.relative_to(root).parts[0]
        except ValueError:
            key = p.stem
        label = key if "n1000" not in p.stem else f"{key}-n1000"
        found.append((p, label))
    if not found:
        raise SystemExit(f"refusing: no lens checkpoints under {root}. "
                         f"Run experiments/fetch_lens.py first; nothing here synthesises one.")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lenses", type=Path, default=Path("lenses"),
                    help="directory to search for lens checkpoints")
    ap.add_argument("--only", default="",
                    help="comma-separated catalogue keys to include (default: all found)")
    ap.add_argument("--live", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    found = discover(a.lenses)
    if a.only.strip():
        want = {k.strip() for k in a.only.split(",") if k.strip()}
        found = [(p, n) for p, n in found if n in want or n.rsplit("-n1000", 1)[0] in want]
        if not found:
            raise SystemExit(f"refusing: none of {sorted(want)} found under {a.lenses}")

    print(f"profiling {len(found)} lens(es) from {a.lenses}", flush=True)
    profiles, t0 = [], time.time()

    def progress(partial, i, n, row):
        name = partial["name"]
        print(f"  {name:<16} layer {row['layer']:>3} ({i}/{n})  a={row['alpha']:>6.3f} "
              f"idE={row['identity_energy']:>5.3f}  PR(J)={row['PR_J']:>8.1f}  "
              f"PR(M)={row['PR_M']:>8.1f}  H2(M)={row['H2_M']:>8.1f}  {row['seconds']:>5.1f}s",
              flush=True)
        if a.live:
            # stream the model in progress alongside the finished ones: at d=4096 a model is
            # minutes, and a page that only repaints per MODEL is not a live page.
            depth_report.render(a.live, profiles + [partial], time.time() - t0, done=False)

    for path, name in found:
        print(f"\n{name}  {path}", flush=True)
        prof = profile_lens(path, name, on_layer=progress)
        profiles.append(prof)
        print(f"  -> peak PR(M)={prof['peak_PR_M']:.1f} at layer {prof['peak_layer']} "
              f"of {prof['n_layers']-1} (relative depth {prof['peak_depth']:.2f})", flush=True)
        if a.live:
            depth_report.render(a.live, profiles, time.time() - t0, done=False)

    if a.live:
        depth_report.render(a.live, profiles, time.time() - t0, done=True)
        print(f"\nlive {a.live.resolve().as_uri()}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"profiles": profiles,
                                     "elapsed_seconds": time.time() - t0}, indent=2))
        print(f"wrote {a.out}", flush=True)

    print(f"\n{'model':<20}{'d':>6}{'layers':>8}{'peak layer':>12}{'rel depth':>11}{'peak PR(M)':>12}")
    for p in sorted(profiles, key=lambda q: q["peak_depth"]):
        print(f"{p['name']:<20}{p['d_model']:>6}{p['n_layers']:>8}{p['peak_layer']:>12}"
              f"{p['peak_depth']:>11.2f}{p['peak_PR_M']:>12.1f}")
    print(trend_summary(profiles))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
