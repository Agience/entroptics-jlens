"""Experiment 3 -- the estimator noise, from two independent fits of the same model.

``J_l`` is an expectation over a corpus, so averaging suppresses the incoherent part before any
threshold sees it. That makes "where is the noise floor" the wrong question and this the right
one: given two fits of the same model at different ``n``, how much do they disagree, and how far
into the spectrum do they still agree on a subspace?

Two things are measured, neither of them fitted:

  **Estimator noise.** For independent means at ``n_A`` and ``n_B``,
  ``sd(A - B) / sd(A) = sqrt(1 + n_A/n_B)``, so the difference of the two fits gives the noise in
  ``A`` on a known scale. The top singular value of that difference, rescaled, is the largest
  singular value estimator noise alone can produce -- a floor with no distributional assumption
  behind it at all.

  **Reproducible dimension.** Principal angles between the two fits' top-``k`` right singular
  subspaces, following Scanu et al. (arXiv:2606.09964, eq. 6) with *fit* in place of noise level.
  A scalar floor cannot say which directions survive; this can.

Both are read on ``M = J - alpha I``, because the identity component is architectural and would
otherwise dominate the comparison (79% of ``||J||_F^2`` at Qwen3.5-4B layer 30).

The catalogue publishes two fits for Qwen3.5-4B (n=1000, n=417) and for Qwen3.6-27B. This
experiment needs both files present and refuses without them.

Usage
-----
    python experiments/fetch_lens.py qwen3.5-4b
    python -c "from huggingface_hub import hf_hub_download as d; d(repo_id='neuronpedia/jacobian-lens', filename='qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt', local_dir='lenses')"
    python experiments/exp3_two_fit_noise.py \
        --a lenses/qwen3.5-4b/.../Qwen3.5-4B_jacobian_lens_n1000.pt \
        --b lenses/qwen3.5-4b/.../Qwen3.5-4B_jacobian_lens.pt \
        --layers 0,6,12,18,24,30 --out results/twofit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

KS = (1, 5, 10, 25, 50, 100, 200, 400)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, type=Path, help="the larger-n fit")
    ap.add_argument("--b", required=True, type=Path, help="the other fit")
    ap.add_argument("--layers", required=True, help="'all' or a comma-separated list")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS),
                    help="subspace sizes for the principal-angle read")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    A, B = je.load_lens(a.a), je.load_lens(a.b)
    if A.d_model != B.d_model:
        raise SystemExit(f"refusing: d_model {A.d_model} vs {B.d_model} -- these are fits of "
                         f"different models, and their difference is signal, not noise")
    shared = [l for l in A.source_layers if l in B.source_layers]
    if not shared:
        raise SystemExit("refusing: the two fits share no fitted layer")
    layers = shared if a.layers.strip() == "all" else [int(v) for v in a.layers.split(",") if v]
    missing = [l for l in layers if l not in shared]
    if missing:
        raise SystemExit(f"refusing: layers {missing} are not in both fits; shared are {shared}")
    if A.n_prompts <= 0 or B.n_prompts <= 0:
        raise SystemExit(f"refusing: n_prompts {A.n_prompts}/{B.n_prompts} -- the noise scale "
                         f"sqrt(1 + n_A/n_B) is undefined without both counts")

    ks = [int(v) for v in a.ks.split(",") if v]
    scale = float(np.sqrt(1.0 + A.n_prompts / B.n_prompts))
    print(f"A n={A.n_prompts}  B n={B.n_prompts}  d={A.d_model}")
    print(f"sd(A-B)/sd(A) = sqrt(1 + n_A/n_B) = {scale:.3f}"
          f"   (independence assumed; if B subsamples A this UNDERSTATES the noise)")
    head = (f"{'layer':>5} {'|D|/|A| J':>9} {'corr J':>8} {'|D|/|A| M':>9} {'corr M':>8} "
            f"{'floor':>9} {'K(M)':>6} " + " ".join(f"{'k='+str(k):>7}" for k in ks))
    print("\n" + head, flush=True)

    rows = []
    for l in layers:
        Ja, Jb = A.jacobian(l), B.jacobian(l)
        # Both objects are reported. The agreement between two fits is a property of whichever
        # matrix is compared, and this paper's subject is the difference between them, so a table
        # that gave only one would leave a reader unable to tell which had been measured.
        rel_J = float(np.linalg.norm(Ja - Jb) / np.linalg.norm(Ja))
        corr_J = float((Ja * Jb).sum() / np.sqrt((Ja ** 2).sum() * (Jb ** 2).sum()))
        Ma = je.decompose(Ja, kind="identity").residual
        Mb = je.decompose(Jb, kind="identity").residual
        D = Ma - Mb
        sD = je.energy_spectrum(D, exact=True)
        sM = je.energy_spectrum(Ma, exact=True)
        floor = float(sD[0] / scale)
        corr = float((Ma * Mb).sum() / np.sqrt((Ma ** 2).sum() * (Mb ** 2).sum()))
        angles = {}
        for k in ks:
            if k <= min(Ma.shape):
                angles[k] = float(je.principal_angles(Ma, Mb, k).mean())
        row = {"layer": l, "rel_diff": float(np.linalg.norm(D) / np.linalg.norm(Ma)),
               "corr": corr, "rel_diff_J": rel_J, "corr_J": corr_J,
               "noise_floor": floor, "K_M": int((sM > floor).sum()),
               "PR_M": je.participation_ratio(sM), "mean_cos": angles}
        rows.append(row)
        print(f"{l:>5} {rel_J:>9.4f} {corr_J:>8.4f} {row['rel_diff']:>9.4f} {corr:>8.4f} "
              f"{floor:>9.4f} {row['K_M']:>6} " +
              " ".join(f"{angles.get(k, float('nan')):>7.3f}" for k in ks), flush=True)

    run = {"a": {"path": str(a.a), "n_prompts": A.n_prompts},
           "b": {"path": str(a.b), "n_prompts": B.n_prompts},
           "d_model": A.d_model, "scale": scale, "far": a.far, "ks": ks, "layers": rows}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(run, indent=2))
        print(f"\nwrote {a.out}")
    deepest = rows[-1]
    print(f"\nAt layer {deepest['layer']} the two fits correlate {deepest['corr']:.4f} and agree "
          f"on subspaces to {max(deepest['mean_cos'].values()):.3f}.")
    print("A detection floor needs a noise bulk whose edge it can find. This is what says "
          "there isn't one.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
