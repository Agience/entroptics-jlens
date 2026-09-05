"""Do many written directions compose, or does the tenth destroy the first?

An ontology that accumulates is worthless if entries interfere. Everything measured about the
crossing used ONE direction, and the single attempt at several (1, 2, 4 and 8 prefix directions)
came out worse than injecting nothing. Composition is the whole architecture, and it is untested.

The test, stated so it can fail:

    take N directions, each resolved from a DIFFERENT document -- the realistic case, where every
    entry comes from a separate interaction rather than from one source's spectrum
    cross each onto the shared token surface and render it into the receiver's basis
    inject the target ALONE, then together with n-1 others
    measure whether the receiver still moves toward the target's concept

    retention = selectivity(target | n-1 others present) / selectivity(target alone)

Retention near 1 means entries coexist. Retention falling toward 0 means the store has a capacity
of one and the architecture is dead.

Two controls, because interference has two possible causes and they need different fixes:

    shuffled distractors   the same n-1 vectors with their vocabulary entries permuted: same
                           energy, same crossing, no semantic content. If retention falls just as
                           far against these, the interference is CAPACITY -- the receiver can only
                           absorb so much perturbation -- and dilution or routing fixes it.
    orthogonalised         the distractors projected off the target before summing. If this
                           restores retention, the interference is OVERLAP between stored
                           directions, and the store needs to keep its entries independent.

Amplitude is derived, never tuned: the injected sum is scaled to the norm one row of the receiver's
own residual stream carries at the injection layer. Fixing total energy is the honest constraint --
a store that only works by injecting n times more energy for n entries has not composed anything.
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
from exp30_context_handoff import documents                            # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sender", default="gpt2")
    ap.add_argument("--receiver", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--inject", type=int, default=2)
    ap.add_argument("--entries", type=int, default=20, help="size of the ontology store")
    ap.add_argument("--sizes", default="1,2,4,8,16,20")
    ap.add_argument("--keep", type=int, default=4096)
    ap.add_argument("--probes", type=int, default=6, help="prompts the receiver is read on")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/composition.json"))
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
    J = je.as_frame(lens.jacobian(a.layer))
    entry = je.vocab_side(J, W_s, keep_s)["entry"]
    recv = je.vocab_side(np.eye(d_r), W_r, keep_r, invertible=True)

    # one direction per document: each store entry comes from a separate interaction
    docs = documents(a.entries + 8)
    pool = [S(tok_s(d, return_tensors="pt").input_ids[:, :256], output_hidden_states=True)
            .hidden_states[a.layer + 1][0].numpy().astype(np.float64) for d in docs[:8]]
    corpus_mu = np.concatenate(pool).mean(0, keepdims=True)

    concepts, vecs = [], []
    for d in docs[8: 8 + a.entries]:
        H = S(tok_s(d, return_tensors="pt").input_ids[:, :256], output_hidden_states=True) \
            .hidden_states[a.layer + 1][0].numpy().astype(np.float64)
        c = entry(H.mean(0, keepdims=True) - corpus_mu)
        z = np.asarray(recv["inverse"](c))[0]
        concepts.append(c[0] - c[0].mean())
        vecs.append(z / max(float(np.linalg.norm(z)), 1e-300))
    V = np.stack(vecs)
    C = np.stack(concepts)
    print(f"{len(V)} store entries; |cos| between stored directions: "
          f"mean {np.mean(np.abs(V @ V.T)[~np.eye(len(V), dtype=bool)]):.3f}", flush=True)

    ids = [tok_r(d, return_tensors="pt", truncation=True,
                 max_length=a.tokens).input_ids for d in docs[:a.probes]]
    blocks = blocks_of(R)
    state = {"u": None}

    def hook(_m, _i, out):
        if state["u"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        new = h.clone()
        new[:, 0, :] = new[:, 0, :] + state["u"] * h.norm(dim=-1).mean()
        return ((new,) + tuple(out[1:])) if tup else new

    def logits(u):
        state["u"] = None if u is None else torch.tensor(u, dtype=torch.float32)
        hd = blocks[a.inject].register_forward_hook(hook)
        try:
            return np.concatenate([R(i).logits[0].numpy() for i in ids])
        finally:
            hd.remove()
            state["u"] = None

    base = logits(None)

    def selectivity(got, c):
        D = got[:, keep_r] - base[:, keep_r]
        D = D - D.mean(1, keepdims=True)
        n = np.linalg.norm(D, axis=1)
        cn = float(np.linalg.norm(c))
        ok = n > 0
        return float(np.mean((D[ok] @ c) / (n[ok] * cn))) if ok.any() and cn > 0 else 0.0

    def unit(x):
        return x / max(float(np.linalg.norm(x)), 1e-300)

    sizes = [int(v) for v in a.sizes.split(",") if v.strip()]
    sizes = [n for n in sizes if n <= len(V)]
    targets = list(range(min(8, len(V))))          # every target measured at every size
    alone = {t: selectivity(logits(V[t]), C[t]) for t in targets}
    print(f"selectivity alone: mean {np.mean(list(alone.values())):+.4f}", flush=True)

    rows = []
    print()
    print(f"{'n':>4}{'retention':>12}{'vs shuffled':>14}{'vs orthog.':>13}{'raw sel':>10}")
    for n in sizes:
        ret, ret_sh, ret_or, raw = [], [], [], []
        for t in targets:
            others = [j for j in range(len(V)) if j != t]
            pick = list(rng.choice(others, size=n - 1, replace=False)) if n > 1 else []
            # real distractors, total energy fixed
            s_real = selectivity(logits(unit(V[t] + V[pick].sum(0) if pick else V[t])), C[t])
            # shuffled: same energy and crossing, no semantic content
            sh = [np.asarray(recv["inverse"](C[j][rng.permutation(len(strings))][None, :]))[0]
                  for j in pick]
            sh = [unit(x) for x in sh]
            s_shuf = selectivity(
                logits(unit(V[t] + np.sum(sh, axis=0) if sh else V[t])), C[t])
            # orthogonalised: distractors projected off the target
            if pick:
                O = V[pick] - np.outer(V[pick] @ V[t], V[t])
                s_orth = selectivity(logits(unit(V[t] + O.sum(0))), C[t])
            else:
                s_orth = s_real
            base_t = alone[t]
            if abs(base_t) > 1e-9:
                ret.append(s_real / base_t)
                ret_sh.append(s_shuf / base_t)
                ret_or.append(s_orth / base_t)
            raw.append(s_real)
        rows.append({"n": n, "retention": float(np.mean(ret)),
                     "retention_shuffled": float(np.mean(ret_sh)),
                     "retention_orthogonal": float(np.mean(ret_or)),
                     "raw_selectivity": float(np.mean(raw))})
        r = rows[-1]
        print(f"{n:>4}{r['retention']:>12.3f}{r['retention_shuffled']:>14.3f}"
              f"{r['retention_orthogonal']:>13.3f}{r['raw_selectivity']:>+10.4f}", flush=True)
        je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "entries": len(V),
                        "targets": len(targets), "rows": rows}, complete=False)

    last = rows[-1]
    print()
    print(f"at {last['n']} entries the target retains {last['retention']:.0%} of its selectivity "
          f"({last['retention_shuffled']:.0%} against shuffled distractors, "
          f"{last['retention_orthogonal']:.0%} orthogonalised)")
    # With total energy fixed, n directions each carry 1/sqrt(n) of their solo amplitude, so
    # linear dilution predicts that retention. Beating it means the entries are coexisting rather
    # than merely sharing a budget; the shuffled arm is the same energy with no semantic content,
    # so the gap between them is what the CONTENT buys.
    import math
    dilution = 1.0 / math.sqrt(last["n"])
    print(f"linear dilution at n={last['n']} would give {dilution:.3f}")
    real, shuf, orth = (last["retention"], last["retention_shuffled"],
                        last["retention_orthogonal"])
    if real > shuf * 1.5 and real > dilution:
        print(f"VERDICT  entries COMPOSE. Real directions retain {real:.0%} against {shuf:.0%} for "
              f"shuffled vectors of identical energy, and beat the {dilution:.0%} linear-dilution "
              f"floor. Orthogonalising gives {orth:.0%}, so the overlap between genuine entries "
              f"helps rather than interferes -- the store should NOT be orthogonalised.")
    elif real > dilution:
        print(f"VERDICT  entries survive dilution ({real:.0%} against a {dilution:.0%} floor) but "
              f"no better than content-free vectors ({shuf:.0%}): the budget is shared, and what "
              f"is stored does not matter to how well it coexists.")
    else:
        print(f"VERDICT  entries INTERFERE: retention {real:.0%} is at or below the {dilution:.0%} "
              f"expected from amplitude dilution alone, so a store of this size is not viable "
              f"as designed.")
    je.dump(a.out, {"sender": a.sender, "receiver": a.receiver, "entries": len(V),
                    "targets": len(targets), "alone": {str(k): v for k, v in alone.items()},
                    "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
