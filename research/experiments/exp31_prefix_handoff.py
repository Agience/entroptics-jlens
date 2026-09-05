"""Hand the receiver tokens it attends to, rather than a perturbation added to its stream.

`exp30` handed one vector across and it closed 3.1% of a 242-token context gap. The sweep said why
that is the ceiling and it is not the crossing: `crossed - shuffled` keeps climbing to +0.174 at
four times the amplitude, six times its value at the peak, while `crossed - floor` turns negative.
More information crosses as amplitude rises; the receiver is destroyed faster than the information
helps. The delivery was the bottleneck.

Adding a vector to a residual stream is not how context enters a model. Context enters as
**positions the model attends to**. So this renders the crossing into the receiver's embedding
space and prepends it as virtual tokens:

    inputs_embeds = [ v_1 ... v_k , embed(anchor) ]

and the receiver reads them with its own attention, at its own scale, with no perturbation to any
computation it performs. Two consequences: nothing is destroyed as `k` or the scale grows, and a
document gets more than one direction. The sender's gist is its residual frame's top `k` right
singular directions rather than a single mean, so `k` virtual tokens carry `k` structures.

Arms are `exp30`'s, so the numbers are comparable:

    floor      the anchor alone
    crossed    k rendered prefix tokens, then the anchor
    shuffled   the same, with each concept's vocabulary entries permuted before rendering
    ceiling    the receiver reads the whole document as tokens
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp26_cross_tokenizer import head_of                              # noqa: E402
from exp30_context_handoff import documents                            # noqa: E402


def embeddings_of(model):
    """The input embedding matrix the prefix vectors have to live beside."""
    return model.get_input_embeddings()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sender", default="gpt2")
    ap.add_argument("--receiver", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--anchor", type=int, default=8)
    ap.add_argument("--continue-tokens", type=int, default=32, dest="cont")
    ap.add_argument("--keep", type=int, default=4096)
    ap.add_argument("--prefix", default="1,2,4,8",
                    help="numbers of virtual tokens to try")
    ap.add_argument("--scales", default="0.5,1.0,2.0",
                    help="prefix norm, in units of the receiver's mean embedding norm")
    ap.add_argument("--out", type=Path, default=Path("results/prefix_handoff.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok_s = transformers.AutoTokenizer.from_pretrained(a.sender)
    tok_r = transformers.AutoTokenizer.from_pretrained(a.receiver)
    S = transformers.AutoModelForCausalLM.from_pretrained(a.sender).eval().float()
    R = transformers.AutoModelForCausalLM.from_pretrained(a.receiver).eval().float()
    lens = je.load_lens(a.lens)
    W_s, W_r = head_of(S), head_of(R)
    d_r = W_r.shape[1]

    v_s, v_r = tok_s.get_vocab(), tok_r.get_vocab()
    shared = sorted(set(v_s) & set(v_r))
    rng = np.random.default_rng(0)
    need = max(a.keep, d_r + 8)
    strings = [shared[i] for i in sorted(rng.choice(len(shared), size=need, replace=False))]
    keep_s = np.array([v_s[t] for t in strings], dtype=int)
    keep_r = np.array([v_r[t] for t in strings], dtype=int)

    emb = embeddings_of(R)
    emb_norm = float(emb.weight.detach().norm(dim=-1).mean())
    prefixes = [int(v) for v in a.prefix.split(",") if v.strip()]
    scales = [float(v) for v in a.scales.split(",") if v.strip()]
    print(f"{a.sender} -> {a.receiver} (d={d_r}); {len(shared)} shared strings; "
          f"mean embedding norm {emb_norm:.3f}", flush=True)

    J = je.as_frame(lens.jacobian(a.layer))
    entry = je.vocab_side(J, W_s, keep_s)["entry"]
    recv = je.vocab_side(np.eye(d_r), W_r, keep_r, invertible=True)
    docs = documents(a.docs)

    pool = []
    for d in docs[: min(8, len(docs))]:
        ids = tok_s(d, return_tensors="pt").input_ids[:, :256]
        pool.append(S(ids, output_hidden_states=True)
                    .hidden_states[a.layer + 1][0].numpy().astype(np.float64))
    corpus_mu = np.concatenate(pool).mean(0, keepdims=True)

    def render(rows):
        """Concept rows on the shared surface -> unit vectors in the receiver's basis."""
        z = np.asarray(recv["inverse"](rows))
        n = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.maximum(n, 1e-300)

    def logprob(prefix_vecs, prompt_ids, cont_ids, scale):
        """Mean log-probability of the continuation, with `prefix_vecs` as leading positions."""
        ids = torch.cat([prompt_ids, cont_ids], dim=1)
        e = emb(ids)
        if prefix_vecs is not None and len(prefix_vecs):
            p = torch.tensor(np.asarray(prefix_vecs), dtype=e.dtype)
            p = (p * scale * emb_norm).unsqueeze(0)
            e = torch.cat([p, e], dim=1)
        lg = R(inputs_embeds=e).logits[0].double()
        lp = torch.log_softmax(lg, dim=-1)
        n_pre = 0 if prefix_vecs is None else len(prefix_vecs)
        n_p = prompt_ids.shape[1] + n_pre
        tgt = ids[0, prompt_ids.shape[1]:]
        return float(lp[n_p - 1: -1].gather(1, tgt[:, None]).squeeze(1).mean())

    rows = []
    for i, doc in enumerate(docs):
        s_ids = tok_s(doc, return_tensors="pt").input_ids[:, :256]
        H = S(s_ids, output_hidden_states=True).hidden_states[a.layer + 1][0].numpy().astype(
            np.float64)
        C = je.as_frame(H - corpus_mu)
        Vt = np.linalg.svd(C, full_matrices=False)[2]          # the document's own directions

        r_ids = tok_r(doc, return_tensors="pt").input_ids
        if r_ids.shape[1] < a.anchor + a.cont + 16:
            continue
        split = r_ids.shape[1] - a.cont
        full, anchor = r_ids[:, :split], r_ids[:, split - a.anchor: split]
        cont = r_ids[:, split:]

        row = {"doc": i, "context_tokens": int(split),
               "floor": logprob(None, anchor, cont, 0.0),
               "ceiling": logprob(None, full, cont, 0.0)}
        for k in prefixes:
            cx = entry(Vt[:k])
            u_cross = render(cx)
            u_shuf = render(cx[:, rng.permutation(len(strings))])
            for sc in scales:
                row[f"crossed@{k}x{sc}"] = logprob(u_cross, anchor, cont, sc)
                row[f"shuffled@{k}x{sc}"] = logprob(u_shuf, anchor, cont, sc)
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{len(docs)} documents", flush=True)

    if not rows:
        raise SystemExit("refusing: no document was long enough to split")

    def col(k):
        return np.array([r[k] for r in rows])

    floor, ceil = col("floor"), col("ceiling")
    gap = ceil - floor
    print()
    print(f"{len(rows)} documents; {np.mean([r['context_tokens'] for r in rows]):.0f} context "
          f"tokens replaced by virtual tokens")
    print()
    print(f"{'k':>4}{'scale':>7}{'crossed':>11}{'shuffled':>11}{'gap closed':>13}{'vs shuf':>10}")
    best = None
    for k in prefixes:
        for sc in scales:
            cr, sh = col(f"crossed@{k}x{sc}"), col(f"shuffled@{k}x{sc}")
            frac = float(np.mean((cr - floor) / np.maximum(gap, 1e-9)))
            print(f"{k:>4}{sc:>7}{cr.mean():>11.4f}{sh.mean():>11.4f}{frac:>12.1%}"
                  f"{(cr - sh).mean():>+10.4f}")
            if best is None or frac > best[2]:
                best = (k, sc, frac)
    print(f"{'floor':>11}{floor.mean():>11.4f}{'':>11}{'0%':>13}")
    print(f"{'ceiling':>11}{ceil.mean():>11.4f}{'':>11}{'100%':>13}")

    k, sc, frac = best
    cr, sh = col(f"crossed@{k}x{sc}"), col(f"shuffled@{k}x{sc}")
    for name, diff in (("shuffled", cr - sh), ("floor", cr - floor)):
        bs = diff[rng.integers(0, len(diff), size=(4000, len(diff)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  crossed - {name:<9} {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")
    print()
    print(f"best: {k} virtual tokens at scale {sc} closes {frac:.1%} of the context gap "
          f"({np.mean([r['context_tokens'] for r in rows]):.0f} tokens -> {k})")
    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "prefixes": prefixes,
                    "scales": scales, "documents": len(rows), "best_k": k, "best_scale": sc,
                    "best_fraction": frac, "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
