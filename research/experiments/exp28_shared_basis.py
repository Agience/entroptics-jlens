"""Stream compression with a well-sampled basis, fitted offline and applied held-out.

`exp24` projected a residual stream onto its own resolved subspace and the model's output did not
survive: top-1 agreement 0.19, and 0.32-0.45 with the token-invariant part preserved. Two things
were wrong with that test, and both make the stream look less compressible than it is.

**It was under-sampled.** The frame was one 128-token sequence against `d=768` -- 128 samples
estimating structure in 768 dimensions. A noise floor on a frame that short is dominated by the
aspect ratio, not by the stream, which is why the resolved rank came out at 5-8 where the same
quantity on concatenated streams reads 17.8-22.1. A rank that small is a statement about the sample
size.

**It fitted the basis on the sequence it then projected.** That is neither the honest test nor the
useful scheme. The scheme that would save compute fits a basis **offline** on many tokens and
applies it **online** to sequences it has never seen.

So: fit on `--fit` prompts, apply to `--test` prompts held out from the fit, and measure the
model's own final top-1 agreement against the unmodified pass. The rank comes from the noise floor
of the well-sampled frame; energy ranks and fixed ranks run beside it.
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
from exp26_cross_tokenizer import blocks_of                            # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--fit", type=int, default=48, help="prompts the basis is fitted on")
    ap.add_argument("--test", type=int, default=6, help="held-out prompts it is applied to")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--layers", default="0,2,4,6,8,10")
    ap.add_argument("--out", type=Path, default=Path("results/shared_basis.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from entroptics.projection import noise_floor
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()
    blocks = blocks_of(model)

    ids_all = wikitext_prompts(tok, a.fit + a.test, a.tokens)
    fit_ids, test_ids = ids_all[:a.fit], ids_all[a.fit:]
    layers = [int(v) for v in a.layers.split(",") if v.strip()]
    print(f"basis fitted on {len(fit_ids)} prompts ({len(fit_ids) * a.tokens} tokens), "
          f"applied to {len(test_ids)} held out", flush=True)

    state = {"mu": None, "V": None}

    def hook(_m, _i, out):
        if state["V"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        c = h.squeeze(0) - state["mu"]
        new = (state["mu"] + (c @ state["V"]) @ state["V"].T).unsqueeze(0)
        return ((new,) + tuple(out[1:])) if tup else new

    rows = []
    print()
    print(f"{'layer':>6}{'T/d':>7}{'K_ent':>7}{'agree@K':>9}{'E99':>6}{'agree@E99':>11}"
          f"{'K=64':>8}{'K=128':>8}")
    for li in layers:
        # --- fit: one frame over every fitted token, so the covariance is well determined
        F = np.concatenate([model(s.unsqueeze(0), output_hidden_states=True)
                            .hidden_states[li + 1][0].numpy().astype(np.float64)
                            for s in fit_ids])
        mu = F.mean(0, keepdims=True)
        C = je.as_frame(F - mu)
        sv = np.linalg.svd(C, full_matrices=False, compute_uv=False)
        Vt = np.linalg.svd(C, full_matrices=False)[2]
        K = max(1, int((sv > float(noise_floor(C, far=a.far, s=sv))).sum()))
        e = np.cumsum(sv ** 2) / float((sv ** 2).sum())
        e99 = int(np.searchsorted(e, 0.99) + 1)
        d = F.shape[1]

        def agree(k):
            got = []
            state["mu"] = torch.tensor(mu, dtype=torch.float32).squeeze(0)
            state["V"] = torch.tensor(Vt[:k].T, dtype=torch.float32)
            handle = blocks[li].register_forward_hook(hook)
            try:
                for s in test_ids:
                    base = None
                    state_V = state["V"]
                    state["V"] = None
                    base = model(s.unsqueeze(0)).logits[0].argmax(-1).numpy()
                    state["V"] = state_V
                    out = model(s.unsqueeze(0)).logits[0].argmax(-1).numpy()
                    got.append(float((out == base).mean()))
            finally:
                handle.remove()
                state["V"] = None
            return float(np.mean(got))

        row = {"layer": li, "d": d, "tokens": len(F), "K_ent": K, "e99": e99,
               "agree_K": agree(K), "agree_e99": agree(e99),
               "agree_64": agree(min(64, d)), "agree_128": agree(min(128, d))}
        rows.append(row)
        print(f"{li:>6}{len(F) / d:>7.1f}{K:>7}{row['agree_K']:>9.3f}{e99:>6}"
              f"{row['agree_e99']:>11.3f}{row['agree_64']:>8.3f}{row['agree_128']:>8.3f}",
              flush=True)
        je.dump(a.out, {"model": a.model, "fit": len(fit_ids), "test": len(test_ids),
                        "layers": rows}, complete=False)

    print()
    print(f"median K from the noise floor: {np.median([r['K_ent'] for r in rows]):.0f} "
          f"of d={rows[0]['d']}  ->  {rows[0]['d'] / max(np.median([r['K_ent'] for r in rows]), 1):.0f}x")
    print(f"median agreement at K: {np.median([r['agree_K'] for r in rows]):.3f}")
    print(f"median agreement at 99% energy "
          f"({np.median([r['e99'] for r in rows]):.0f} dirs): "
          f"{np.median([r['agree_e99'] for r in rows]):.3f}")
    je.dump(a.out, {"model": a.model, "fit": len(fit_ids), "test": len(test_ids),
                    "layers": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
