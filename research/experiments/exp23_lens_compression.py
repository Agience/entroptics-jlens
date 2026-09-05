"""What the coupling buys: a Jacobian lens applied at a fraction of its cost, at the same read.

A Jacobian lens costs a $d^2$ matmul per token per layer. Two entroptics reads change that, and
neither is available from the lens alone:

    identity decomposition   J = alpha*I + M, an exact orthogonal projection onto span(I).
                             The identity is FULL RANK, so it cannot be truncated -- but it is
                             also free, a scalar multiply. Only M needs a matrix.
    derived rank             the number of directions M resolves above its own noise floor, K,
                             obtained without labels, without the downstream task, and without
                             running the lens.

Together they give a compressed lens

    J h  ~=  alpha * h  +  M_K h        cost  d + 2dK   against   d^2

and the question this measures is whether the compressed lens produces the SAME READ: the same
top-1 token out of the unembedding, at the same positions.

Three things are compared, because the claim is only worth stating if the entroptics rank beats
what you would pick without it:

    entroptics     K from the noise floor of M
    energy         the smallest rank holding 90% / 99% of ||M||_F^2
    no-decomposition   truncating J itself to rank K, which has to spend its rank representing
                       the identity before it can carry anything else

The last is what quantifies the decomposition: same budget, spent on the wrong basis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp4_stream_complement import (collect_streams, unembed_fn,       # noqa: E402
                                    wikitext_prompts)


def top1(logits):
    return np.asarray(logits).argmax(1)


def energy_rank(sv, frac):
    """Smallest rank holding ``frac`` of the squared Frobenius energy."""
    e = np.cumsum(sv ** 2) / max(float((sv ** 2).sum()), 1e-300)
    return int(np.searchsorted(e, frac) + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--target", type=float, default=0.99,
                    help="top-1 agreement with the full lens that counts as the same read")
    ap.add_argument("--out", type=Path, default=Path("results/lens_compression.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    lens = je.load_lens(a.lens)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    ids = wikitext_prompts(tok, a.prompts, a.tokens, skip=a.skip)
    streams, _ = collect_streams(m, ids)
    un = unembed_fn(m)
    d = lens.d_model
    print(f"{len(streams)} prompts, d={d}, target agreement {a.target:.0%}")

    rows = []
    print()
    print(f"{'layer':>6}{'K_ent':>7}{'agree':>8}{'k*':>5}{'E90':>5}{'E99':>5}"
          f"{'J-rank needed':>15}{'FLOP saving':>13}")
    for l in lens.source_layers:
        H = np.concatenate([s[l + 1] for s in streams])
        J = je.as_frame(lens.jacobian(l))
        dec = je.decompose(J, kind="identity")
        M, alpha = dec.residual, dec.alpha
        ref = top1(un(H @ J.T))

        K_ent = int(je.transport_spectrum(M, far=a.far).K)
        sv = np.linalg.svd(M, compute_uv=False)
        e90, e99 = energy_rank(sv, 0.90), energy_rank(sv, 0.99)

        def agree_M(k):
            if k < 1:
                return float((top1(un(alpha * H)) == ref).mean())
            Mk, _ = je.truncated_pair(M, min(k, d))
            return float((top1(un(alpha * H + H @ Mk.T)) == ref).mean())

        def agree_J(k):
            Jk, _ = je.truncated_pair(J, min(k, d))
            return float((top1(un(H @ Jk.T)) == ref).mean())

        # smallest rank on M reaching the target, by bisection over a doubling ladder
        ladder = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, d) if k <= d]
        kstar = next((k for k in ladder if agree_M(k) >= a.target), None)
        if kstar and kstar > 1:                       # refine between the last two rungs
            lo = ladder[ladder.index(kstar) - 1]
            while lo + 1 < kstar:
                mid = (lo + kstar) // 2
                if agree_M(mid) >= a.target:
                    kstar = mid
                else:
                    lo = mid
        # the rank raw J needs for the same agreement, same ladder
        jstar = next((k for k in ladder if agree_J(k) >= a.target), None)

        ag = agree_M(K_ent)
        cost_full = d * d
        cost_ent = d + 2 * d * K_ent
        rows.append({"layer": l, "K_entroptics": K_ent, "agreement_at_K": ag,
                     "k_star": kstar, "energy90": e90, "energy99": e99,
                     "j_rank_needed": jstar, "alpha": float(alpha),
                     "flop_ratio": cost_full / cost_ent})
        print(f"{l:>6}{K_ent:>7}{ag:>8.3f}{str(kstar):>5}{e90:>5}{e99:>5}"
              f"{str(jstar):>15}{cost_full / cost_ent:>12.1f}x", flush=True)
        je.dump(a.out, {"model": a.model, "d_model": d, "prompts": len(streams),
                        "target": a.target, "layers": rows}, complete=False)

    ok = [r for r in rows if r["agreement_at_K"] >= a.target]
    print()
    print(f"entroptics rank reaches the target on {len(ok)}/{len(rows)} layers")
    print(f"median agreement at K_entroptics: "
          f"{np.median([r['agreement_at_K'] for r in rows]):.3f}")
    print(f"median FLOP saving at K_entroptics: "
          f"{np.median([r['flop_ratio'] for r in rows]):.1f}x")
    ks = [r["k_star"] for r in rows if r["k_star"]]
    js = [r["j_rank_needed"] for r in rows if r["j_rank_needed"]]
    if ks and js:
        print(f"median rank needed on M: {np.median(ks):.0f}   on raw J: {np.median(js):.0f}   "
              f"({np.median(js) / max(np.median(ks), 1):.1f}x more without the decomposition)")
    print(f"median energy-99 rank: {np.median([r['energy99'] for r in rows]):.0f}   "
          f"energy-90: {np.median([r['energy90'] for r in rows]):.0f}")
    je.dump(a.out, {"model": a.model, "d_model": d, "prompts": len(streams),
                    "target": a.target, "layers": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
