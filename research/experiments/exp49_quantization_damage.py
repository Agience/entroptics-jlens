"""Which layers did quantization break? The reads applied to ordinary weights, not to a lens.

Every other script here reads a fitted Jacobian lens. This one reads a model's own attention
output projections, because nothing in `principal_angles`, `participation_ratio` or `decompose` is
lens-specific -- they are linear algebra on a matrix, and a `(d, d)` weight matrix is a matrix.

The question is one people answer today with an eval set: a model was quantized, is it still good?
That returns one number for the whole model and says nothing about where the damage is. Comparing
the weights directly is exact, costs seconds, needs no data, and is per layer.

    python experiments/exp49_quantization_damage.py
    python experiments/exp49_quantization_damage.py --model gpt2 --bits 8,4,3 --k 64

Measured on gpt2 and quoted in the README: int8 holds above 0.99 everywhere except layer 11 at
0.9621, and int4 falls unevenly from 0.8669 at layer 0 to 0.4429 at layer 10. Eight cells are
undefined -- int4 at layers 1 and 11, int3 at layers 1, 2, 3, 8, 10 and 11 -- because the quantized
matrix has fewer than 64 directions above round-off or a run of equal singular values across the
64th, so it has no top-64 subspace to compare (``agreement``). Layer 11's effective rank collapses
from 27.2 to 4.2.

Scope: these numbers say what changed in the weights. Whether a given agreement score predicts
task loss is not measured here and is the experiment that would make this a decision procedure
rather than a measurement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def quantize(W: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric per-tensor quantization, the crudest thing anyone actually ships.

    Deliberately not a good quantizer. The point is to produce a known, tunable amount of damage
    and ask whether the read sees it, not to evaluate a quantization scheme.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.abs(W).max()) / qmax
    if scale == 0.0:
        raise ValueError("the weight matrix is identically zero; there is nothing to quantize")
    return np.round(W / scale).clip(-qmax - 1, qmax) * scale


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


def agreement(W: np.ndarray, q: np.ndarray, k: int) -> tuple[float | None, str | None]:
    """Mean cosine between the top-``k`` subspaces of ``W`` and its quantization, or ``None`` and
    the reason where the quantized matrix has no top-``k`` subspace.

    Heavy quantization rounds most weights to zero and leaves a few repeated levels, so the
    quantized matrix can have fewer than ``k`` directions above round-off, or a run of equal
    singular values across the ``k``-th. ``principal_angles`` refuses both, and the refusal is
    the measurement: any number there would be whichever basis LAPACK happened to return."""
    try:
        return float(je.principal_angles(W, q, k=k).mean()), None
    except ValueError as exc:
        return None, str(exc).removeprefix("principal_angles: ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--bits", default="8,4,3", help="bit widths to compare against the original")
    ap.add_argument("--k", type=int, default=64, help="directions to compare per layer")
    ap.add_argument("--out", type=Path, default=Path("results/quantization_damage.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()
    blocks = model.transformer.h
    widths = [int(b) for b in a.bits.split(",")]

    print(f"{a.model}: {len(blocks)} layers, attn.c_proj, top-{a.k} directions\n")
    head = "".join(f"{'int' + str(b):>12}" for b in widths)
    print(f"{'layer':>6}{'PR(W)':>9}{head}{'PR(int' + str(min(widths)) + ')':>12}")

    rows = []
    for l, blk in enumerate(blocks):
        W = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        pr = je.participation_ratio(je.energy_spectrum(W))
        agree, undefined = {}, {}
        for b in widths:
            agree[b], why = agreement(W, quantize(W, b), a.k)
            if why is not None:
                undefined[b] = why
        pr_worst = je.participation_ratio(je.energy_spectrum(quantize(W, min(widths))))
        rows.append({"layer": l, "pr": pr, "agreement": agree, "undefined": undefined,
                     "pr_worst_bits": pr_worst})
        cells = "".join(f"{'undefined':>12}" if agree[b] is None else f"{agree[b]:>12.4f}"
                        for b in widths)
        print(f"{l:>6}{pr:>9.1f}{cells}{pr_worst:>12.1f}", flush=True)
        je.dump(a.out, {"model": a.model, "bits": widths, "k": a.k, "layers": rows},
                complete=False)

    # Does effective rank PREDICT which layers break? Reported because the answer is "partly",
    # and a partial predictor stated as a rule is how a screening heuristic gets trusted.
    # Over the layers where the score is defined at that width; a layer whose quantized matrix has
    # no top-k subspace has no score to rank, and is named rather than ranked.
    worst = min(widths)
    kept = [r for r in rows if r["agreement"][worst] is not None]
    print(f"\nint{worst} is undefined at k={a.k} on layers "
          f"{[r['layer'] for r in rows if r['agreement'][worst] is None]}")
    pr_all = [r["pr"] for r in kept]
    dmg = [r["agreement"][worst] for r in kept]
    rho = spearman(pr_all, dmg) if len(kept) > 2 else None
    k = max(1, len(kept) // 4)
    low_pr = set(np.argsort(pr_all)[:k])
    hurt = set(np.argsort(dmg)[:k])
    print(f"Spearman(PR, int{worst} agreement) over {len(kept)} layers = "
          + ("undefined" if rho is None else f"{rho:+.3f}"))
    print(f"the {k} lowest-rank layers and the {k} worst-damaged overlap on "
          f"{len(low_pr & hurt)} of {k}.")
    print("A moderate signal, not a rule. The direct before/after comparison is exact and costs")
    print("seconds, so measure the damage rather than predicting it.")

    je.dump(a.out, {"model": a.model, "bits": widths, "k": a.k, "layers": rows,
                    "spearman_pr_vs_damage": rho}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
