"""The compressible object is the stream, not the transport -- tested causally.

`exp23` looked for the efficiency win in the Jacobian matrix and found none: reproducing the
lens's top-1 read takes 706-737 of 768 available ranks, so `J` is not low-rank in any way that
pays. But `J` was the wrong target. The residual stream resolves **17.8-22.1 directions above its
own noise floor at gpt2's d=768** (sec 7.2) -- a factor of 35 between the space the stream lives in
and the structure it carries.

If that structure is what the model computes with, projecting the stream onto it should leave the
model's output alone; if the discarded directions matter, the output moves. Nothing else in this
work intervenes (sec 8a), so this is also the first read here with a causal answer rather than a
descriptive one.

    intervene   replace the residual stream at layer l with its rank-K projection, H V_K V_K^T,
                and run the remaining layers on it
    measure     top-1 agreement of the model's FINAL logits against the unmodified forward pass
    choose K    from the entroptics noise floor -- no labels, no output comparison, no search

and against the baselines you would use without the instrument: the energy ranks holding 90% and
99% of the frame, and a fixed fraction of `d`. The instrument earns its place only if its rank
preserves the output where a cheaper rule does not, or reaches the same output at a smaller rank.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp4_stream_complement import wikitext_prompts                    # noqa: E402


def blocks(model):
    """The decoder blocks, whose outputs are the residual stream."""
    for path in ("transformer.h", "model.layers", "model.language_model.layers", "gpt_neox.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj, path
    raise ValueError("could not locate the decoder blocks on this model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/stream_compression.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
    if next(model.parameters()).dtype != torch.float32:
        model = model.float()
    hs, path = blocks(model)
    ids = wikitext_prompts(tok, a.prompts, a.tokens, skip=a.skip)
    print(f"{len(ids)} prompts, {len(hs)} blocks at {path}")

    state = {"basis": None}

    def hook(_mod, _inp, out):
        """Replace the block's residual output with its rank-K projection."""
        V = state["basis"]
        if V is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        proj = (h.squeeze(0) @ V) @ V.T
        proj = proj.unsqueeze(0)
        return (proj,) + tuple(out[1:]) if isinstance(out, tuple) else proj

    rows = []
    print()
    print(f"{'layer':>6}{'K_ent':>7}{'d/K':>7}{'agree@K_ent':>13}{'E90':>6}{'agree@E90':>11}"
          f"{'E99':>6}{'agree@E99':>11}")
    for li in range(len(hs) - 1):                       # the last block feeds the norm directly
        per = {k: [] for k in ("K", "e90", "e99")}
        agree = {k: [] for k in ("K", "e90", "e99")}
        for seq in ids:
            x = seq.unsqueeze(0)
            base = model(x).logits[0].argmax(-1).numpy()
            h = model(x, output_hidden_states=True).hidden_states[li + 1][0]
            Hn = h.numpy().astype(np.float64)
            U, sv, Vt = np.linalg.svd(je.as_frame(Hn), full_matrices=False)
            floor = float(je.noise_floor_of(Hn, far=a.far)) if hasattr(je, "noise_floor_of") \
                else float(__import__("entroptics.projection", fromlist=["noise_floor"])
                           .noise_floor(je.as_frame(Hn), far=a.far, s=sv))
            K = max(1, int((sv > floor).sum()))
            e = np.cumsum(sv ** 2) / float((sv ** 2).sum())
            ranks = {"K": K, "e90": int(np.searchsorted(e, 0.90) + 1),
                     "e99": int(np.searchsorted(e, 0.99) + 1)}
            for name, k in ranks.items():
                per[name].append(k)
                state["basis"] = torch.tensor(Vt[:k].T, dtype=h.dtype)
                handle = hs[li].register_forward_hook(hook)
                try:
                    got = model(x).logits[0].argmax(-1).numpy()
                finally:
                    handle.remove()
                    state["basis"] = None
                agree[name].append(float((got == base).mean()))
        row = {"layer": li, **{f"rank_{k}": float(np.mean(v)) for k, v in per.items()},
               **{f"agree_{k}": float(np.mean(v)) for k, v in agree.items()},
               "d": int(h.shape[-1])}
        row["compression"] = row["d"] / max(row["rank_K"], 1e-9)
        rows.append(row)
        print(f"{li:>6}{row['rank_K']:>7.1f}{row['compression']:>7.1f}{row['agree_K']:>13.3f}"
              f"{row['rank_e90']:>6.0f}{row['agree_e90']:>11.3f}{row['rank_e99']:>6.0f}"
              f"{row['agree_e99']:>11.3f}", flush=True)
        je.dump(a.out, {"model": a.model, "prompts": len(ids), "layers": rows}, complete=False)

    print()
    print(f"median agreement at the entroptics rank: "
          f"{np.median([r['agree_K'] for r in rows]):.3f}  "
          f"at {np.median([r['compression'] for r in rows]):.0f}x compression")
    print(f"median agreement at energy-90 ({np.median([r['rank_e90'] for r in rows]):.0f} dirs): "
          f"{np.median([r['agree_e90'] for r in rows]):.3f}")
    print(f"median agreement at energy-99 ({np.median([r['rank_e99'] for r in rows]):.0f} dirs): "
          f"{np.median([r['agree_e99'] for r in rows]):.3f}")
    je.dump(a.out, {"model": a.model, "prompts": len(ids), "layers": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
