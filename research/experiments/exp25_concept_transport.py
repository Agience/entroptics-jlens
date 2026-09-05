"""Carry a direction one model resolved into another model, and act with it.

Every read before this one scores a crossing: coverage, certify, transfer and realise all return a
number saying how well two sides agree. A screen is not for scoring. It is a shared basis two
systems both convert onto, so a structure resolved on one side can be **rendered out on the other**
-- ``entry`` in, ``inverse`` out. What comes back is not a verdict about the receiver. It is a
vector in the receiver's own residual basis, carrying a structure the sender found, that the
receiver has never been shown.

Two models of different residual width share no residual basis, but if their token ids agree they
share a readout basis, so the vocabulary is where the crossing happens (``vocab_side``).

    resolve     the leading direction of ``M = J - alpha I`` at layer ``l`` of the SENDER, the
                direction its transport carries hardest
    cross       enter it on the shared token basis through the sender's own readout
    render      leave through the RECEIVER's inverse: a vector in the receiver's residual basis
    act         add it to the receiver's stream and run the receiver forward

The measurement is the shift in the receiver's own logits over the tokens the sender's direction
names -- the sender names a target in its vocabulary, the receiver is scored on whether it moves
there.

Three controls, because a steering vector that moves logits proves nothing on its own:

    random      a random direction of the same norm
    shuffled    the concept's vocabulary entries permuted before rendering: same energy, same
                crossing, no correspondence between direction and tokens
    raw         the sender's vector injected directly, with no crossing. Both models here are 768
                wide, so this is possible -- and it is the comparison that says whether the
                vocabulary crossing carries meaning a shared width does not.
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
    """The output head matrix, as ``(vocab, d)``."""
    import torch
    for path in ("lm_head.weight", "transformer.wte.weight"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj.detach().to(dtype=torch.float64).numpy()
    raise ValueError("no output head found on this model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sender", default="gpt2")
    ap.add_argument("--receiver", default="distilgpt2")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--layer", type=int, default=6, help="sender layer to take the direction from")
    ap.add_argument("--inject", type=int, default=2, help="receiver block to add it to")
    ap.add_argument("--keep", type=int, default=4096, help="size of the shared token sub-basis")
    ap.add_argument("--alpha", type=float, default=1.0, help="injection strength, in stream norms")
    ap.add_argument("--top", type=int, default=32, help="tokens the concept is scored on")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/concept_transport.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.sender)
    tok_r = transformers.AutoTokenizer.from_pretrained(a.receiver)
    if tok.get_vocab() != tok_r.get_vocab():
        raise ValueError(f"{a.sender} and {a.receiver} do not share a token basis; the crossing "
                         f"has nowhere to happen")
    S = transformers.AutoModelForCausalLM.from_pretrained(a.sender).eval().float()
    R = transformers.AutoModelForCausalLM.from_pretrained(a.receiver).eval().float()
    lens = je.load_lens(a.lens)
    W_s, W_r = head_of(S), head_of(R)
    d_s, d_r = W_s.shape[1], W_r.shape[1]
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(W_s.shape[0], size=max(a.keep, d_r + 8), replace=False))
    print(f"sender {a.sender} d={d_s} layer {a.layer} | receiver {a.receiver} d={d_r} "
          f"block {a.inject} | shared basis {len(keep)} tokens", flush=True)

    # resolve: the direction the sender's transport carries hardest, identity removed
    J = je.as_frame(lens.jacobian(a.layer))
    M = je.decompose(J, kind="identity").residual
    v = np.linalg.svd(M, full_matrices=False)[2][0]

    # cross: onto the shared token basis, through the sender's own readout
    sender = je.vocab_side(J, W_s, keep)
    concept = sender["entry"](v[None, :])

    # render: out through the receiver's inverse. Identity transport = the receiver's own readout
    recv = je.vocab_side(np.eye(d_r), W_r, keep, invertible=True)

    def render(c):
        z = np.asarray(recv["inverse"](c))[0]
        n = float(np.linalg.norm(z))
        return z / n if n > 0 else z

    u_cross = render(concept)
    u_shuf = render(concept[:, rng.permutation(len(keep))])
    u_rand = rng.standard_normal(d_r)
    u_rand = u_rand / np.linalg.norm(u_rand)
    u_raw = v / np.linalg.norm(v) if d_s == d_r else None

    target = keep[np.argsort(concept[0])[-a.top:]]
    print(f"concept names: {tok.decode(target[-8:])!r}", flush=True)

    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    blocks = R.transformer.h
    state = {"u": None}

    def hook(_m, _i, out):
        if state["u"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        scale = a.alpha * h.norm(dim=-1).mean()
        new = h + scale * state["u"]
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
    arms = [("crossed", u_cross), ("shuffled", u_shuf), ("random", u_rand)]
    if u_raw is not None:
        arms.append(("raw-no-crossing", u_raw))

    # Selectivity, not magnitude. Any perturbation inflates logits, and the concept's largest
    # entries are the frequent tokens, so a mean shift over its top tokens scores loudness --
    # measured, a random direction wins that. What the transport claims is that the receiver moves
    # in the direction the sender NAMED, so the statistic is the correlation between the induced
    # logit change and the concept profile, per position, both centred. Scale-free by construction.
    c = concept[0] - concept[0].mean()
    cn = float(np.linalg.norm(c))

    def selectivity(got):
        D = got[:, keep] - base[:, keep]
        D = D - D.mean(1, keepdims=True)
        n = np.linalg.norm(D, axis=1)
        ok = n > 0
        return float(np.mean((D[ok] @ c) / (n[ok] * cn))) if ok.any() and cn > 0 else 0.0

    shifts = {}
    for name, u in arms:
        shifts[name] = selectivity(run(u))

    print()
    print(f"{'arm':>18}{'selectivity':>14}{'over random':>14}")
    for name, _ in arms:
        mark = "" if name == "random" else f"{shifts[name] - shifts['random']:>+14.4f}"
        print(f"{name:>18}{shifts[name]:>+14.4f}{mark}")

    print()
    print(f"crossing over shuffled: {shifts['crossed'] - shifts['shuffled']:+.4f}")
    if "raw-no-crossing" in shifts:
        print(f"crossing over raw     : "
              f"{shifts['crossed'] - shifts['raw-no-crossing']:+.4f}")
    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "layer": a.layer,
                    "inject": a.inject, "keep": len(keep), "alpha": a.alpha,
                    "target_tokens": [int(t) for t in target], "shifts": shifts},
            complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
