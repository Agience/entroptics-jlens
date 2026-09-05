"""Does the crossing carry *which* structure, or merely some structure?

`exp26` transported the leading direction of gpt2's transport into pythia-70m -- no shared width,
no shared tokenizer -- and the receiver moved toward the tokens the sender named: selectivity
+0.147 against -0.190 for the same concept shuffled, difference +0.336 with a tight interval.

One reading of that is weaker than it looks. The leading direction names function words -- ` as`,
` is`, ` for`, `.`, `,` -- so it is largely a frequency direction, and a frequency direction might
survive any sensible mapping between two language models. The shuffled control shows that token
identity matters; it does not show that the crossing distinguishes *this* structure from *that*
one.

The test that does: cross several sender directions, and score every one of them against every
concept. If the crossing carries identity, direction `i` moves the receiver toward concept `i` and
not toward concept `j`, and the matrix is diagonal. If it carries only "some structure", every row
looks the same and the off-diagonal is as strong as the diagonal.

    rows        the sender's singular directions of M = J - alpha I, crossed and rendered
    columns     the concepts those directions name, on the shared string surface
    entry       selectivity of the receiver's induced logit change for that (direction, concept)

The number that matters is the gap between the diagonal and the best off-diagonal entry in each
row: whether the receiver moved toward what it was sent rather than toward what it was not.
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
from exp26_cross_tokenizer import blocks_of, head_of                   # noqa: E402


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
    ap.add_argument("--components", default="0,1,2,3,4",
                    help="which singular directions of M to transport")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/transport_specificity.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok_s = transformers.AutoTokenizer.from_pretrained(a.sender)
    tok_r = transformers.AutoTokenizer.from_pretrained(a.receiver)
    v_s, v_r = tok_s.get_vocab(), tok_r.get_vocab()
    S = transformers.AutoModelForCausalLM.from_pretrained(a.sender).eval().float()
    R = transformers.AutoModelForCausalLM.from_pretrained(a.receiver).eval().float()
    lens = je.load_lens(a.lens)
    W_s, W_r = head_of(S), head_of(R)
    d_r = W_r.shape[1]

    shared = sorted(set(v_s) & set(v_r))
    rng = np.random.default_rng(0)
    need = max(a.keep, d_r + 8)
    pick = sorted(rng.choice(len(shared), size=need, replace=False))
    strings = [shared[i] for i in pick]
    keep_s = np.array([v_s[t] for t in strings], dtype=int)
    keep_r = np.array([v_r[t] for t in strings], dtype=int)
    comps = [int(v) for v in a.components.split(",") if v.strip()]
    print(f"{a.sender} -> {a.receiver}; {len(shared)} shared strings, crossing on {len(strings)}; "
          f"components {comps}", flush=True)

    J = je.as_frame(lens.jacobian(a.layer))
    M = je.decompose(J, kind="identity").residual
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    entry = je.vocab_side(J, W_s, keep_s)["entry"]
    recv = je.vocab_side(np.eye(d_r), W_r, keep_r, invertible=True)

    concepts, vectors = [], []
    for i in comps:
        c = entry(Vt[i][None, :])[0]
        concepts.append(c - c.mean())
        z = np.asarray(recv["inverse"](c[None, :]))[0]
        n = float(np.linalg.norm(z))
        vectors.append(z / n if n > 0 else z)
        top = [strings[j] for j in np.argsort(c)[-5:]]
        safe = [t.encode("unicode_escape").decode("ascii") for t in top]
        print(f"  component {i:>2} names {safe}", flush=True)

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

    def selectivity(got, c):
        D = got[:, keep_r] - base[:, keep_r]
        D = D - D.mean(1, keepdims=True)
        n = np.linalg.norm(D, axis=1)
        cn = float(np.linalg.norm(c))
        ok = n > 0
        return float(np.mean((D[ok] @ c) / (n[ok] * cn))) if ok.any() and cn > 0 else 0.0

    Mx = np.zeros((len(comps), len(comps)))
    for r, u in enumerate(vectors):
        got = run(u)
        for c_i, c in enumerate(concepts):
            Mx[r, c_i] = selectivity(got, c)
        print(f"  sent {comps[r]:>2}: " + "  ".join(f"{v:+.3f}" for v in Mx[r]), flush=True)

    print()
    print("selectivity matrix -- rows: direction sent, columns: concept scored against")
    print("       " + "".join(f"{c:>9}" for c in comps))
    for r, c in enumerate(comps):
        print(f"sent {c:>2}" + "".join(f"{v:>9.3f}" for v in Mx[r]))

    diag = np.diag(Mx)
    off = Mx.copy()
    np.fill_diagonal(off, -np.inf)
    best_off = off.max(1)
    wins = int((diag > best_off).sum())
    print()
    print(f"diagonal beats every off-diagonal in its row on {wins}/{len(comps)} rows")
    print(f"mean diagonal {diag.mean():+.4f}   mean best off-diagonal {best_off.mean():+.4f}   "
          f"gap {diag.mean() - best_off.mean():+.4f}")
    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "components": comps,
                    "shared_strings": len(shared), "crossing_size": len(strings),
                    "matrix": Mx.tolist(), "diagonal_wins": wins,
                    "mean_diagonal": float(diag.mean()),
                    "mean_best_off": float(best_off.mean())}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
