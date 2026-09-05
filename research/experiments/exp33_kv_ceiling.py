"""How compressible is a KV cache, before any rule is built to compress it.

Five efficiency attempts failed here, and the procedural lesson from the last one is the only part
worth carrying forward: **measure the ceiling first**. Early exit was investigated for three
experiments before anyone asked what a perfect oracle would save, and the answer was 3.7% -- there
was never anything for a rule to win.

So this asks the ceiling question about a KV cache and nothing else. No entroptics rule, no derived
threshold, no comparison. Just: how much of a cache can be discarded before the model's output
moves, measured in the setting that would deploy it.

    prefix      run the model over a context, keep its KV cache
    compress    replace each layer's keys and values by their rank-r approximation over positions
    decode      run the continuation against the compressed cache
    measure     top-1 agreement with the uncompressed run, per rank

The sanity check is the point of the design: at full rank the compression is the identity, so
agreement **must** be 1.000. A KV intervention that cannot reproduce the model when it is a no-op
would make every number below a statement about the intervention -- which is exactly the bug that
made the early-exit experiment unreadable for three runs.

What comes out is a curve: agreement against the fraction of the cache kept. The knee of that curve
is the ceiling on any KV compression method, entroptics or otherwise, and it is worth knowing
before building one.
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


def cache_tensors(cache):
    """(keys, values) per layer, for both the legacy tuple cache and the Cache object."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    if hasattr(cache, "layers"):
        lays = [l for l in cache.layers if hasattr(l, "keys") and l.keys is not None]
        if not lays:
            raise ValueError(
                "this model's cache holds no keys/values -- it is a linear or hybrid attention "
                "architecture whose state is already a fixed-size summary rather than a per-"
                "position cache. KV compression is a claim about quadratic attention and does not "
                "apply here.")
        if len(lays) != len(cache.layers):
            raise ValueError(f"hybrid cache: {len(lays)} of {len(cache.layers)} layers hold "
                             f"keys/values. Compressing only some of them measures a mixture, "
                             f"not a ceiling; run this on a uniformly quadratic model.")
        return ([l.keys for l in lays], [l.values for l in lays])
    ks = [kv[0] for kv in cache]
    vs = [kv[1] for kv in cache]
    return ks, vs


def set_cache(cache, ks, vs):
    """Write compressed tensors back into a cache object of whichever shape it is."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i, (k, v) in enumerate(zip(ks, vs)):
            cache.key_cache[i], cache.value_cache[i] = k, v
        return cache
    if hasattr(cache, "layers"):
        for lay, k, v in zip(cache.layers, ks, vs):
            lay.keys, lay.values = k, v
        return cache
    return tuple((k, v) for k, v in zip(ks, vs))


def evict(k_t, v_t, keep_idx):
    """Keep only the positions in ``keep_idx``; this is what KV compression actually does.

    An earlier version of this took a rank-``r`` SVD of each (T, D) cache slab. With T=96 and D=64
    the rank was capped by the HEAD DIMENSION, so it measured whether the head dim can be halved --
    a different question, and the answer was no. A KV cache grows along POSITIONS, and every
    deployed method (eviction, windowing, summarisation) trades against that axis.
    """
    return k_t[:, :, keep_idx, :], v_t[:, :, keep_idx, :]


def attention_mass(model, pre):
    """Total attention each prefix position receives, summed over layers and heads.

    Used to build the oracle: the strongest simple eviction keeps the most-attended positions. A
    ceiling measured against this is a ceiling against a well-informed method, not a straw one.
    """
    out = model(pre, use_cache=True, output_attentions=True)
    if not getattr(out, "attentions", None):
        raise ValueError("the model returned no attention weights; load it with "
                         "attn_implementation='eager' or the oracle has nothing to rank by")
    mass = None
    for att in out.attentions:                       # (B, H, q, k)
        m = att[0].sum(dim=0).sum(dim=0).double()    # attention RECEIVED by each key position
        mass = m if mass is None else mass + m
    return mass.numpy(), out.past_key_values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2",
                    help="must be quadratic attention: Qwen3.5 is linear/hybrid "
                         "and has no per-position KV cache to compress")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--prefix", type=int, default=96)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--window", type=int, default=0,
                    help="always keep the last W positions. Deployed eviction "
                         "(H2O, SnapKV) keeps a recent window plus heavy hitters; "
                         "keeping only heavy hitters is a stricter rule than anyone "
                         "ships, and measuring that ceiling answers a question "
                         "nobody asked.")
    ap.add_argument("--ranks", default="4,8,16,24,32,48,64,80",
                    help="numbers of prefix POSITIONS to keep, most-attended first")
    ap.add_argument("--out", type=Path, default=Path("results/kv_ceiling.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    # eager attention: SDPA never materialises the weights, so `output_attentions` comes
    # back None and the oracle has nothing to rank positions by.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, attn_implementation="eager").eval().float()
    ids = wikitext_prompts(tok, a.prompts, a.prefix + a.decode)
    ranks = [int(v) for v in a.ranks.split(",") if v.strip()]
    print(f"{a.model}: {len(ids)} prompts, {a.prefix}-token prefix, {a.decode} decoded",
          flush=True)

    def run(seq, k):
        """Decode against a cache holding only the `k` most-attended prefix positions."""
        pre, post = seq[: a.prefix].unsqueeze(0), seq[a.prefix:].unsqueeze(0)
        mass, cache = attention_mass(model, pre)
        ks, vs = cache_tensors(cache)
        shape = ks[0].shape
        if k is not None and k < shape[2]:
            T = shape[2]
            win = np.arange(max(0, T - max(a.window, 1)), T)     # the recent window, always kept
            room = max(k - len(win), 0)
            heavy = np.argsort(mass)[::-1]
            heavy = np.array([i for i in heavy if i not in set(win.tolist())][:room], dtype=int)
            top = np.unique(np.concatenate([win, heavy])) if room else win
            idx = torch.tensor(top, dtype=torch.long)
            cache = set_cache(cache, *zip(*[evict(kk, vv, idx) for kk, vv in zip(ks, vs)]))
        # Position ids must be the token's TRUE positions, not derived from cache length.
        # gpt2 adds absolute position embeddings, so a cache shortened from 640 to 512 tells the
        # next token it sits at 512 when it sits at 640, and every decoded token gets the wrong
        # encoding. The full-retention sanity check cannot see this: with nothing evicted there is
        # no length mismatch, so the bug only exists in exactly the runs being measured.
        pos = torch.arange(a.prefix, a.prefix + post.shape[1], dtype=torch.long).unsqueeze(0)
        return (model(post, past_key_values=cache, position_ids=pos).logits[0].numpy(), shape)

    base, shape = run(ids[0], None)
    ident, _ = run(ids[0], 10 ** 6)                # keep everything: the eviction is the identity
    agree = float((base.argmax(1) == ident.argmax(1)).mean())
    if agree < 0.9999:
        raise ValueError(f"the KV intervention is not a no-op at full retention: agreement "
                         f"{agree:.4f}. Every number below would be about the intervention.")
    # Full retention cannot detect a positional mismatch, because nothing is shortened. Dropping
    # the SINGLE least-attended position shortens the cache by one while removing almost no
    # information: if that collapses the output, the eviction mechanism is broken rather than the
    # cache being incompressible. This check is here because its absence let a position-id bug
    # produce a plausible ceiling curve twice.
    one, _ = run(ids[0], int(shape[2]) - 1)
    a1 = float((base.argmax(1) == one.argmax(1)).mean())
    if a1 < 0.9:
        raise ValueError(f"dropping the least-attended position of {shape[2]} changed "
                         f"{1 - a1:.0%} of the decoded tokens. That is the eviction mechanism, not "
                         f"the cache: check position ids and the recent window.")
    B, H, T, D = shape
    print(f"SANITY  full retention {agree:.4f}  drop-one {a1:.4f}  |  cache per layer: "
          f"{H} heads x {T} positions x {D} dims", flush=True)

    rows = []
    print()
    print(f"{'keep':>6}{'kept':>9}{'agreement':>12}{'d-NLL':>12}{'ppl x':>11}")
    for r in ranks + [T]:
        got, dnll = [], []
        for seq in ids:
            lg, _ = run(seq, r)
            b, _ = run(seq, None)
            got.append(float((lg.argmax(1) == b.argmax(1)).mean()))
            # perplexity is what the literature reports; exact top-1 is stricter and counts
            # disagreements that change no answer. Both are shown so neither can be cherry-picked.
            tgt = seq[a.prefix + 1:].numpy()
            n = min(len(tgt), lg.shape[0] - 1)
            def nll(x):
                z = x[:n] - x[:n].max(1, keepdims=True)
                return float(-(z[np.arange(n), tgt[:n]] - np.log(np.exp(z).sum(1))).mean())
            dnll.append(nll(lg) - nll(b))
        frac = min(r, T) / T
        rows.append({"rank": int(r), "kept": frac, "agreement": float(np.mean(got)),
                     "delta_nll": float(np.mean(dnll)),
                     "ppl_ratio": float(np.exp(np.mean(dnll)))})
        print(f"{r:>6}{frac:>8.1%}{np.mean(got):>12.3f}{np.mean(dnll):>+12.4f}"
              f"{np.exp(np.mean(dnll)):>11.3f}", flush=True)
        je.dump(a.out, {"model": a.model, "prefix": a.prefix, "decode": a.decode,
                        "heads": int(H), "positions": int(T), "head_dim": int(D),
                        "rows": rows}, complete=False)

    print()
    for name, ok in (("99% top-1 agreement",
                      [r for r in rows if r["agreement"] >= 0.99]),
                     ("1% perplexity increase",
                      [r for r in rows if r["ppl_ratio"] <= 1.01]),
                     ("5% perplexity increase",
                      [r for r in rows if r["ppl_ratio"] <= 1.05])):
        if ok:
            b = min(ok, key=lambda r: r["kept"])
            print(f"  {name:<24} needs {b['kept']:>6.1%} of the cache "
                  f"-> {1 - b['kept']:.1%} discardable")
        else:
            print(f"  {name:<24} not reached at any retention swept")
    je.dump(a.out, {"model": a.model, "prefix": a.prefix, "decode": a.decode,
                    "heads": int(H), "positions": int(T), "head_dim": int(D),
                    "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
