"""A crossing between two models that share neither a tokenizer nor a width.

`exp25` carried a direction from gpt2 into distilgpt2. Both are 768 wide and share a vocabulary, so
the vector could also be injected directly -- and the crossing beat that direct injection, which
was the point. But it left the mechanism's actual claim untested: that two systems meet by *each
having a conversion onto a common surface*, with no correspondence between their internals.

gpt2 and pythia-70m-deduped share nothing. Different residual widths (768 against 512), so a vector
from one cannot be placed in the other at all. Different tokenizers, so their vocabularies are
different objects with different ids. What they do share is **strings**: many byte-pair tokens
spell the same text in both. That set of strings is a surface both can convert onto, and it is the
only route between them.

    surface     the token strings common to both vocabularies, held at each model's own ids
    sender      gpt2's residual -> its transport -> its head, restricted to those ids
    receiver    pythia's head restricted to its ids, inverted -> pythia's residual
    act         add the rendered vector to pythia's stream and run pythia forward

Building the surface *is* the ontology match: two vocabularies constructed independently, aligned
on nothing but the text they spell, with no paired training and no learned mapping.

The measurement is selectivity -- the correlation between the receiver's induced logit change and
the profile the sender named, both centred, so a perturbation that merely inflates logits scores
zero. Controls are the same concept with its entries shuffled (same energy, same crossing, no
correspondence) and a random direction. There is no direct-injection control here, because there
cannot be one: the widths differ. That is the case the mechanism claims.
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


def head_of(model):
    """The output head, as ``(vocab, d)``."""
    import torch
    for path in ("lm_head.weight", "embed_out.weight", "transformer.wte.weight",
                 "gpt_neox.embed_out.weight"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj.detach().to(dtype=torch.float64).numpy()
    raise ValueError("no output head found on this model")


def blocks_of(model):
    """The decoder blocks, whose outputs are the residual stream."""
    for path in ("transformer.h", "gpt_neox.layers", "model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("could not locate the decoder blocks")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sender", default="gpt2")
    ap.add_argument("--receiver", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--inject", type=int, default=2)
    ap.add_argument("--keep", type=int, default=4096)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples over positions")
    ap.add_argument("--out", type=Path, default=Path("results/cross_tokenizer.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok_s = transformers.AutoTokenizer.from_pretrained(a.sender)
    tok_r = transformers.AutoTokenizer.from_pretrained(a.receiver)
    v_s, v_r = tok_s.get_vocab(), tok_r.get_vocab()
    if v_s == v_r:
        raise ValueError("these models share a tokenizer; use exp25 for that case")

    S = transformers.AutoModelForCausalLM.from_pretrained(a.sender).eval().float()
    R = transformers.AutoModelForCausalLM.from_pretrained(a.receiver).eval().float()
    lens = je.load_lens(a.lens)
    W_s, W_r = head_of(S), head_of(R)
    d_s, d_r = W_s.shape[1], W_r.shape[1]

    # --- the ontology match: strings both vocabularies spell, at each model's own ids
    shared = sorted(set(v_s) & set(v_r))
    rng = np.random.default_rng(0)
    need = max(a.keep, d_r + 8)
    if len(shared) < need:
        raise ValueError(f"only {len(shared)} token strings are common to both vocabularies; "
                         f"{need} are needed for an invertible crossing")
    pick = rng.choice(len(shared), size=need, replace=False)
    strings = [shared[i] for i in sorted(pick)]
    keep_s = np.array([v_s[t] for t in strings], dtype=int)
    keep_r = np.array([v_r[t] for t in strings], dtype=int)
    print(f"sender {a.sender} d={d_s} | receiver {a.receiver} d={d_r}", flush=True)
    print(f"vocabularies: {len(v_s)} and {len(v_r)} tokens, {len(shared)} strings in common; "
          f"crossing on {len(strings)}", flush=True)

    # --- resolve, cross, render
    J = je.as_frame(lens.jacobian(a.layer))
    M = je.decompose(J, kind="identity").residual
    v = np.linalg.svd(M, full_matrices=False)[2][0]
    concept = je.vocab_side(J, W_s, keep_s)["entry"](v[None, :])
    recv = je.vocab_side(np.eye(d_r), W_r, keep_r, invertible=True)

    def render(c):
        z = np.asarray(recv["inverse"](c))[0]
        n = float(np.linalg.norm(z))
        return z / n if n > 0 else z

    u_cross = render(concept)
    u_shuf = render(concept[:, rng.permutation(len(strings))])
    u_rand = rng.standard_normal(d_r)
    u_rand = u_rand / np.linalg.norm(u_rand)

    # The matched control: a direction the lens did NOT resolve, carried by the same pipeline and
    # scored against its own profile. `shuffled` and `random` both break the round trip before it
    # starts, so they can only score at or below zero whatever the crossing carries; neither
    # separates "this direction is structure" from "the pseudo-inverse round trip returns what was
    # put into it". Only an arm that completes the round trip with an arbitrary payload does that.
    w = rng.standard_normal(d_s)
    w = w / np.linalg.norm(w)
    concept_ctl = je.vocab_side(J, W_s, keep_s)["entry"](w[None, :])
    u_ctl = render(concept_ctl)
    top = [strings[i] for i in np.argsort(concept[0])[-8:]]
    # ascii-safe: byte-pair strings carry markers (U+0120 for a leading space) that a console
    # codepage cannot encode, and a crossing must not die on its own progress print.
    safe = [t.encode("unicode_escape").decode("ascii") for t in top]
    print(f"concept names: {safe}", flush=True)

    ids = wikitext_prompts(tok_r, a.prompts, a.tokens)
    blocks = blocks_of(R)
    state = {"u": None}

    def hook(_m, _i, out):
        if state["u"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        new = h + a.alpha * h.norm(dim=-1).mean() * state["u"]
        return ((new,) + tuple(out[1:])) if tup else new

    def run(u):
        state["u"] = None if u is None else torch.tensor(u, dtype=torch.float32)
        handle = blocks[a.inject].register_forward_hook(hook)
        try:
            return np.concatenate([R(s.unsqueeze(0)).logits[0].numpy() for s in ids])
        finally:
            handle.remove()
            state["u"] = None

    base = run(None)

    def centred(profile):
        c = profile[0] - profile[0].mean()
        return c, float(np.linalg.norm(c))

    def per_position(got, target):
        c, cn = target
        D = got[:, keep_r] - base[:, keep_r]
        D = D - D.mean(1, keepdims=True)
        n = np.linalg.norm(D, axis=1)
        out = np.zeros(len(D))
        ok = n > 0
        out[ok] = (D[ok] @ c) / (n[ok] * cn)
        return out

    sent, ctl = centred(concept), centred(concept_ctl)
    arms = {name: per_position(run(u), tgt) for name, u, tgt in (
        ("crossed", u_cross, sent),
        ("shuffled", u_shuf, sent),
        ("random", u_rand, sent),
        ("unresolved", u_ctl, ctl),
    )}

    print()
    print(f"{'arm':>10}{'selectivity':>14}{'95% CI':>22}")
    stats = {}
    for name, vals in arms.items():
        idx = rng.integers(0, len(vals), size=(a.boot, len(vals)))
        bs = vals[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        stats[name] = {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi)}
        print(f"{name:>10}{vals.mean():>+14.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}")

    print()
    for other in ("shuffled", "random", "unresolved"):
        diff = arms["crossed"] - arms[other]
        idx = rng.integers(0, len(diff), size=(a.boot, len(diff)))
        bs = diff[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "significant" if lo > 0 or hi < 0 else "not significant"
        stats[f"crossed_minus_{other}"] = {"mean": float(diff.mean()),
                                           "lo": float(lo), "hi": float(hi)}
        print(f"crossed - {other:<9} {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "d_sender": int(d_s),
                    "d_receiver": int(d_r), "vocab_sender": len(v_s), "vocab_receiver": len(v_r),
                    "shared_strings": len(shared), "crossing_size": len(strings),
                    "layer": a.layer, "inject": a.inject, "alpha": a.alpha,
                    "positions": int(len(arms["crossed"])), "stats": stats}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
