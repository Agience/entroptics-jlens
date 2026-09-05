"""One vector instead of a context: what a crossing is worth in tokens.

sec 7.8 showed a structure resolved in one model arriving in another that shares no width and no
tokenizer, and steering it selectively. That is a logit correlation, and nobody buys a logit
correlation. This attaches a task.

    A reads a document. B never sees it. B is asked to continue the text anyway.

The handoff is one vector. A's residual stream over the document is summarised as its mean
direction, centred against the corpus mean so what crosses is *this document* rather than the shape
of English; that vector is entered on the shared string surface through A's readout and rendered
into B's residual basis through B's inverse. B is given only a short syntactic anchor -- the last
few tokens -- plus the vector, and scored on the true continuation.

Four arms fix the scale, so the result is a fraction rather than a number with no referent:

    floor      B sees the anchor alone. What B can do with no context.
    crossed    B sees the anchor plus the rendered vector. The claim.
    shuffled   B sees the anchor plus the same vector with its vocabulary entries permuted.
               Same energy, same crossing, no correspondence: the control.
    ceiling    B sees the whole document as tokens. What the context is worth when B reads it.

The number that means something is the fraction of the floor-to-ceiling gap the crossing closes,
against the token count it replaces. If a 96-token context is worth a certain lift and one vector
recovers a quarter of it, that is a stated exchange rate between a vector and a prefill.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp26_cross_tokenizer import blocks_of, head_of                   # noqa: E402


def documents(n, chars=1200):
    """Plain wikitext passages, as text -- the two models tokenise them differently."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out, buf = [], []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        buf.append(t)
        joined = "\n\n".join(buf)
        if len(joined) >= chars:
            out.append(joined[:chars])
            buf = []
            if len(out) >= n:
                break
    if len(out) < n:
        raise SystemExit(f"refusing: wikitext yielded {len(out)} passages, asked for {n}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sender", default="gpt2")
    ap.add_argument("--receiver", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--layer", type=int, default=6, help="sender layer the gist is read from")
    ap.add_argument("--inject", type=int, default=2, help="receiver block the vector enters")
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--anchor", type=int, default=8, help="tokens of context B is given")
    ap.add_argument("--continue-tokens", type=int, default=32, dest="cont")
    ap.add_argument("--keep", type=int, default=4096)
    ap.add_argument("--alphas", default="0.02,0.05,0.1,0.2,0.5",
                    help="injection strengths to sweep, in stream norms. A single "
                         "strength was the first mistake here: at 2.0 the injection "
                         "destroys the receiver and the transfer is unmeasurable.")
    ap.add_argument("--first-only", action="store_true", dest="first_only",
                    help="inject at the first position only, rather than every one")
    ap.add_argument("--out", type=Path, default=Path("results/context_handoff.json"))
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
    print(f"{a.sender} (d={W_s.shape[1]}) -> {a.receiver} (d={d_r}); "
          f"{len(shared)} shared strings, crossing on {len(strings)}", flush=True)

    J = je.as_frame(lens.jacobian(a.layer))
    entry = je.vocab_side(J, W_s, keep_s)["entry"]
    recv = je.vocab_side(np.eye(d_r), W_r, keep_r, invertible=True)
    alphas = [float(v) for v in a.alphas.split(",") if v.strip()]
    docs = documents(a.docs)

    # the corpus mean of A's stream: what crosses must be THIS document, not the shape of English
    pool = []
    for d in docs[: min(8, len(docs))]:
        ids = tok_s(d, return_tensors="pt").input_ids[:, :256]
        pool.append(S(ids, output_hidden_states=True)
                    .hidden_states[a.layer + 1][0].numpy().astype(np.float64))
    corpus_mu = np.concatenate(pool).mean(0, keepdims=True)

    blocks = blocks_of(R)
    state = {"u": None, "alpha": 0.0}

    def hook(_m, _i, out):
        if state["u"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        add = state["alpha"] * h.norm(dim=-1).mean() * state["u"]
        if a.first_only:
            new = h.clone()
            new[:, 0, :] = new[:, 0, :] + add
        else:
            new = h + add
        return ((new,) + tuple(out[1:])) if tup else new

    def logprob(prompt_ids, cont_ids, u, alpha=0.0):
        """Mean log-probability B assigns to the true continuation."""
        ids = torch.cat([prompt_ids, cont_ids], dim=1)
        state["u"] = None if u is None else torch.tensor(u, dtype=torch.float32)
        state["alpha"] = alpha
        handle = blocks[a.inject].register_forward_hook(hook)
        try:
            lg = R(ids).logits[0].double()
        finally:
            handle.remove()
            state["u"] = None
        lp = torch.log_softmax(lg, dim=-1)
        n_p = prompt_ids.shape[1]
        tgt = ids[0, n_p:]
        got = lp[n_p - 1: -1].gather(1, tgt[:, None]).squeeze(1)
        return float(got.mean())

    rows = []
    for i, doc in enumerate(docs):
        s_ids = tok_s(doc, return_tensors="pt").input_ids[:, :256]
        H = S(s_ids, output_hidden_states=True).hidden_states[a.layer + 1][0].numpy().astype(
            np.float64)
        gist = H.mean(0, keepdims=True) - corpus_mu
        concept = entry(gist)
        z = np.asarray(recv["inverse"](concept))[0]
        u_cross = z / max(float(np.linalg.norm(z)), 1e-300)
        zs = np.asarray(recv["inverse"](concept[:, rng.permutation(len(strings))]))[0]
        u_shuf = zs / max(float(np.linalg.norm(zs)), 1e-300)

        r_ids = tok_r(doc, return_tensors="pt").input_ids
        if r_ids.shape[1] < a.anchor + a.cont + 16:
            continue
        split = r_ids.shape[1] - a.cont
        full, anchor = r_ids[:, :split], r_ids[:, split - a.anchor: split]
        cont = r_ids[:, split:]

        row = {"doc": i, "context_tokens": int(split), "anchor_tokens": int(a.anchor),
               "floor": logprob(anchor, cont, None),
               "ceiling": logprob(full, cont, None)}
        for al in alphas:
            row[f"crossed@{al}"] = logprob(anchor, cont, u_cross, al)
            row[f"shuffled@{al}"] = logprob(anchor, cont, u_shuf, al)
        rows.append(row)
        if (i + 1) % 6 == 0:
            print(f"  {i + 1}/{len(docs)} documents", flush=True)

    if not rows:
        raise SystemExit("refusing: no document was long enough to split")

    def col(k):
        return np.array([r[k] for r in rows])

    floor, ceil = col("floor"), col("ceiling")
    gap = ceil - floor
    print()
    print(f"{len(rows)} documents; context replaced "
          f"{np.mean([r['context_tokens'] for r in rows]):.0f} tokens with 1 vector")
    print()
    print(f"{'alpha':>8}{'crossed':>12}{'shuffled':>12}{'gap closed':>13}{'vs shuffled':>14}")
    best = None
    for al in alphas:
        cr, sh = col(f"crossed@{al}"), col(f"shuffled@{al}")
        frac = float(np.mean((cr - floor) / np.maximum(gap, 1e-9)))
        print(f"{al:>8}{cr.mean():>12.4f}{sh.mean():>12.4f}{frac:>12.1%}"
              f"{(cr - sh).mean():>+14.4f}")
        if best is None or cr.mean() > best[1]:
            best = (al, float(cr.mean()), frac)
    print(f"{'floor':>8}{floor.mean():>12.4f}{'':>12}{'0%':>13}")
    print(f"{'ceiling':>8}{ceil.mean():>12.4f}{'':>12}{'100%':>13}")
    print()
    print(f"best strength {best[0]}: closes {best[2]:.1%} of the context gap")

    al = best[0]
    crossed, shuf = col(f"crossed@{al}"), col(f"shuffled@{al}")
    d_cs = crossed - shuf
    boot = rng.integers(0, len(d_cs), size=(4000, len(d_cs)))
    lo, hi = np.percentile(d_cs[boot].mean(1), [2.5, 97.5])
    d_cf = crossed - floor
    lo2, hi2 = np.percentile(d_cf[rng.integers(0, len(d_cf), size=(4000, len(d_cf)))].mean(1),
                             [2.5, 97.5])
    print(f"at alpha={al}:")
    print(f"  crossed - shuffled  {d_cs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")
    print(f"  crossed - floor     {d_cf.mean():+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]  "
          f"{'significant' if lo2 > 0 or hi2 < 0 else 'not significant'}")

    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "layer": a.layer,
                    "inject": a.inject, "alphas": alphas, "best_alpha": best[0], "anchor": a.anchor,
                    "documents": len(rows), "rows": rows,
                    "crossed_minus_shuffled": {"mean": float(d_cs.mean()),
                                               "lo": float(lo), "hi": float(hi)}},
            complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
