"""P6: does the agreement score predict what quantising a layer costs the model? It does not.

`exp49` measures how much of each layer's map survives quantisation. That is exact and says
nothing about whether you will care. This asks the question that would make it a decision
procedure: quantise ONE layer, measure the loss it actually costs, and correlate.

    python experiments/exp50_quantization_cost.py
    python experiments/exp50_quantization_cost.py --model gpt2 --bits 4 --chunks 12

Measured on gpt2, 2026-09-02, baseline loss 4.1575 over 12 chunks of 1024 tokens:

    Spearman(agreement, loss increase) = +0.070      Pearson = +0.265

No relationship, and the extremes invert. Layer 0 has the HIGHEST agreement (0.8669) and the
HIGHEST cost (+0.368). Layer 1 has the lowest agreement (0.3783) and costs almost nothing
(+0.007). The score says which weights moved; it does not say which weights mattered.

This is the fifth independent attempt to turn one of these reads into a decision and the fifth
failure -- P1 escalation, P2 abstention, P3 probe placement, the complement read, and this. The
reads measure geometry precisely and geometry has not predicted behaviour in any test put to it.
Kept and shipped because a negative result is a deliverable: it stops the next person spending a
week on a mixed-precision heuristic that does not work.

The evaluation text defaults to this repository's own README.md, so the run needs no dataset
download and is deterministic. It is a small and unrepresentative corpus, which matters for the
absolute loss and not for the correlation, since every layer is scored on the same text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp49_quantization_damage import quantize, spearman               # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--k", type=int, default=64, help="directions compared per layer")
    ap.add_argument("--chunks", type=int, default=12, help="1024-token chunks to score on")
    ap.add_argument("--text", type=Path, default=Path("README.md"))
    ap.add_argument("--out", type=Path, default=Path("results/quantization_cost.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()

    raw = a.text.read_text(encoding="utf-8")[20000:120000]
    ids = tok(raw, return_tensors="pt").input_ids[0]
    n = min(a.chunks, ids.shape[0] // 1024)
    if n < 2:
        raise ValueError(f"{a.text} gives only {ids.shape[0]} tokens; need at least 2048 to score")
    chunks = ids[: n * 1024].view(-1, 1024)

    def loss_of(m) -> float:
        total = 0.0
        for c in chunks:
            total += float(m(c.unsqueeze(0), labels=c.unsqueeze(0)).loss)
        return total / len(chunks)

    base = loss_of(model)
    print(f"{a.model}: baseline loss {base:.4f} over {n} chunks of 1024 tokens\n")
    print(f"{'layer':>6}{'agreement':>11}{'loss increase':>15}")

    rows = []
    for l, blk in enumerate(model.transformer.h):
        proj = blk.attn.c_proj
        original = proj.weight.detach().clone()
        W = original.numpy().astype(np.float64)
        q = quantize(W, a.bits)
        agree = float(je.principal_angles(W, q, k=a.k).mean())
        # Swapped in, scored, and put back. The restore is in the loop body rather than a finally
        # because a failure here should leave the model visibly wrong rather than quietly patched.
        proj.weight.data = torch.tensor(q, dtype=original.dtype)
        cost = loss_of(model) - base
        proj.weight.data = original
        rows.append({"layer": l, "agreement": agree, "loss_increase": cost})
        print(f"{l:>6}{agree:>11.4f}{cost:>15.4f}", flush=True)
        je.dump(a.out, {"model": a.model, "bits": a.bits, "baseline_loss": base,
                        "layers": rows}, complete=False)

    agree = [r["agreement"] for r in rows]
    cost = [r["loss_increase"] for r in rows]
    rho = spearman(agree, cost)
    pearson = float(np.corrcoef(agree, cost)[0, 1])
    k = max(1, len(rows) // 4)
    predicted = set(int(i) for i in np.argsort(agree)[:k])          # lowest agreement
    actual = set(int(i) for i in np.argsort(cost)[-k:])             # highest cost
    print(f"\nSpearman = {rho:+.3f}   Pearson = {pearson:+.3f}")
    print(f"lowest agreement: layer {int(np.argmin(agree))}   "
          f"most costly: layer {int(np.argmax(cost))}")
    print(f"the {k} lowest-agreement layers and the {k} most costly overlap on "
          f"{len(predicted & actual)} of {k}.")
    print("\nThe score does not predict the cost. It says which weights moved, not which")
    print("mattered. Use it to describe what a transformation changed, not to allocate bits.")

    je.dump(a.out, {"model": a.model, "bits": a.bits, "baseline_loss": base, "layers": rows,
                    "spearman": rho, "pearson": pearson,
                    "top_k_overlap": [len(predicted & actual), k]}, complete=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.LensFormatError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
