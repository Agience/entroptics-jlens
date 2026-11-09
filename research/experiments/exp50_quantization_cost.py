"""P6: does the agreement score predict what quantizing a layer costs the model? It does not.

`exp49` measures how much of each layer's map survives quantization. That is exact and says
nothing about whether you will care. This asks the question that would make it a decision
procedure: quantize ONE layer, measure the loss it actually costs, and correlate.

    python experiments/exp50_quantization_cost.py
    python experiments/exp50_quantization_cost.py --model gpt2 --bits 4 --chunks 12

Measured on gpt2 at int4, baseline loss 3.3169 over 12 wikitext-2 test sequences of 1024 tokens,
over the ten layers whose agreement is defined (layers 1 and 11 have no top-64 subspace at int4;
see ``exp49_quantization_damage.agreement``):

    Spearman(agreement, loss increase) = -0.139      Pearson = +0.384

No relationship, and the extremes do not line up. Layer 0 has the HIGHEST agreement (0.8669) and
the HIGHEST cost (+0.481). Layer 10 has the lowest agreement (0.4429) and costs +0.044, less than
four of the other nine. The score says which weights moved; it does not say which weights mattered.

This is the fifth independent attempt to turn one of these reads into a decision and the fifth
failure -- P1 escalation, P2 abstention, P3 probe placement, the complement read, and this. The
reads measure geometry precisely and geometry has not predicted behaviour in any test put to it.
Kept and shipped because a negative result is a deliverable: it stops the next person spending a
week on a mixed-precision heuristic that does not work.

The evaluation text is the leading wikitext-2 test sequences, the corpus family the published
lenses were fitted on, so every run scores the same tokens. ``--text`` scores a file instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp49_quantization_damage import agreement, quantize, spearman    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--k", type=int, default=64, help="directions compared per layer")
    ap.add_argument("--chunks", type=int, default=12, help="1024-token chunks to score on")
    ap.add_argument("--text", type=Path, default=None,
                    help="score on this file instead of wikitext-2 (the default)")
    ap.add_argument("--out", type=Path, default=Path("results/quantization_cost.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()

    if a.text is None:
        # A fixed corpus: the wikitext-2 test sequences the lenses' fit family is drawn from, the
        # same leading ones on every run, so a rerun scores the same tokens.
        from exp4_stream_complement import wikitext_prompts
        chunks = torch.stack(wikitext_prompts(tok, a.chunks, 1024))
        n = int(chunks.shape[0])
    else:
        raw = a.text.read_text(encoding="utf-8")
        ids = tok(raw, return_tensors="pt").input_ids[0]
        n = min(a.chunks, ids.shape[0] // 1024)
        if n < 2:
            raise ValueError(f"{a.text} gives only {ids.shape[0]} tokens; need at least 2048")
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
        agree, why = agreement(W, q, a.k)
        # Swapped in, scored, and put back. The restore is in the loop body rather than a finally
        # because a failure here should leave the model visibly wrong rather than quietly patched.
        proj.weight.data = torch.tensor(q, dtype=original.dtype)
        cost = loss_of(model) - base
        proj.weight.data = original
        rows.append({"layer": l, "agreement": agree, "undefined": why, "loss_increase": cost})
        shown = f"{'undefined':>11}" if agree is None else f"{agree:>11.4f}"
        print(f"{l:>6}{shown}{cost:>15.4f}", flush=True)
        je.dump(a.out, {"model": a.model, "bits": a.bits, "baseline_loss": base,
                        "layers": rows}, complete=False)

    # Ranked over the layers whose score is defined; a layer with no top-k subspace has no score.
    kept = [r for r in rows if r["agreement"] is not None]
    print(f"\nundefined at k={a.k}: layers {[r['layer'] for r in rows if r['agreement'] is None]}")
    layers = [r["layer"] for r in kept]
    agree = [r["agreement"] for r in kept]
    cost = [r["loss_increase"] for r in kept]
    rho = spearman(agree, cost)
    pearson = float(np.corrcoef(agree, cost)[0, 1])
    k = max(1, len(kept) // 4)
    predicted = set(layers[int(i)] for i in np.argsort(agree)[:k])  # lowest agreement
    actual = set(layers[int(i)] for i in np.argsort(cost)[-k:])     # highest cost
    print(f"over {len(kept)} layers: Spearman = {rho:+.3f}   Pearson = {pearson:+.3f}")
    print(f"lowest agreement: layer {layers[int(np.argmin(agree))]}   "
          f"most costly: layer {layers[int(np.argmax(cost))]}")
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
