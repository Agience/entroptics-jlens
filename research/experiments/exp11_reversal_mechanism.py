"""Hypothesis 3 for the fit-ranking reversal: threshold instability in basis selection.

`coverage` picks each basis size from a noise floor. At high k_t the basis includes directions
sitting near that threshold, and which side of it they land on can differ between two fits for
reasons unrelated to fit quality -- so the ranking could invert without coverage itself being at
fault.

Test: force both fits to the SAME basis size and re-rank. If the reversals vanish, the mechanism
is basis selection, and the fix is to pin k when comparing two readouts.

Two mechanisms are already ruled out: the fits differing most at those
layers, and the noisier fit carrying extra spurious directions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def cov_fixed(sig_svd_Vt, img, k_sig, k_read):
    """Coverage with both bases pinned to a given size -- no threshold anywhere."""
    Vs = sig_svd_Vt[:k_sig].T
    Vt = np.linalg.svd(je.as_frame(img), full_matrices=False)[2][:k_read].T
    c = np.clip(np.linalg.svd(Vs.T @ Vt, compute_uv=False), 0.0, 1.0)
    return float((c ** 2).sum() / k_sig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/reversal_mechanism.json"))
    a = ap.parse_args(argv)

    A = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens_n1000.pt")
    B = je.load_lens("lenses/qwen3.5-4b/jlens/Salesforce-wikitext/"
                     "Qwen3.5-4B_jacobian_lens.pt")
    z = np.load("streams/qwen35_4b_streams.npz")
    S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                 key=lambda k: int(k[1:]))]
    rows = []
    print(f"{'layer':>6}{'floor-picked':>14}{'fixed-k':>11}{'k_t A':>7}{'k_t B':>7}{'flip':>6}",
          flush=True)
    for l in A.source_layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        ea, eb, fa, fb, kta, ktb = [], [], [], [], [], []
        for s in S:
            H = s[l + 1]
            Ha, Hb = H @ Ja.T, H @ Jb.T
            ca, cb = je.coverage(H, Ha), je.coverage(H, Hb)
            ea.append(ca.excess)
            eb.append(cb.excess)
            kta.append(ca.k_readout)
            ktb.append(cb.k_readout)
            Vt_sig = np.linalg.svd(je.as_frame(H), full_matrices=False)[2]
            k = min(ca.k_readout, cb.k_readout)
            fa.append(cov_fixed(Vt_sig, Ha, ca.k_signal, k))
            fb.append(cov_fixed(Vt_sig, Hb, ca.k_signal, k))
        d_floor = float(np.mean(ea) - np.mean(eb))
        d_fixed = float(np.mean(fa) - np.mean(fb))
        flip = (d_floor < 0) != (d_fixed < 0)
        rows.append({"layer": l, "d_floor": d_floor, "d_fixed": d_fixed,
                     "kt_a": float(np.mean(kta)), "kt_b": float(np.mean(ktb)), "flip": flip})
        print(f"{l:>6}{d_floor:>+14.5f}{d_fixed:>+11.5f}{np.mean(kta):>7.1f}"
              f"{np.mean(ktb):>7.1f}{'YES' if flip else '-':>6}", flush=True)
        je.dump(a.out, {"rows": rows}, complete=False)

    n = len(rows)
    w_floor = sum(1 for r in rows if r["d_floor"] > 0)
    w_fixed = sum(1 for r in rows if r["d_fixed"] > 0)
    rev = [r for r in rows if r["d_floor"] < 0]
    fixed_rev = sum(1 for r in rev if r["d_fixed"] > 0)
    print()
    print(f"floor-picked basis : better fit wins {w_floor}/{n}")
    print(f"fixed-k basis      : better fit wins {w_fixed}/{n}")
    print(f"of the {len(rev)} reversals, {fixed_rev} become correct under a pinned basis")
    je.dump(a.out, {"rows": rows}, complete=True)
    verdict = ("basis selection explains it" if fixed_rev >= max(1, len(rev) - 1)
               else "basis selection does NOT explain it")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults,
            ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
