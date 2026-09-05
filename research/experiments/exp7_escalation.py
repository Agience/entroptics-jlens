"""P1 -- escalation. Can the small model tell you, by itself, when to call the large one?

This is the first benchmark that is about the product rather than the instrument, and it is an
ordinary supervised evaluation: there is a label, a score, and baselines everyone already uses.

    label   at each token position, does the small model's top-1 prediction match the large
            model's? That is exactly the decision an escalation policy is trying to make -- if
            they agree, the large call was wasted.
    scores  computed from the SMALL model alone, since a score needing the large model's
            forward pass saves nothing.
    metric  AUROC, plus cost saved at fixed agreement, which is what a deployment actually buys.

Baselines, both standard and both free:

    max-softmax   the model's own top probability. The default confidence signal everywhere.
    negentropy    negated predictive entropy over the vocabulary.

The screen-based score:

    resolved     the fraction of a position's activation energy lying inside the subspace the
                 stream resolves above its own noise floor. The hypothesis is that positions
                 sitting mostly outside the resolved sector are the ones where the small model
                 is improvising, and improvised positions are where it diverges from the large
                 model.

A read that does not beat max-softmax here is not worth deploying, whatever else it measures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp4_stream_complement import collect_streams, wikitext_prompts   # noqa: E402


def auroc(y, s):
    """Rank-based AUROC. No sklearn dependency for one statistic."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties, else a score with ties is scored as if it ordered them
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def resolved_energy_fraction(H, far: float = 0.05):
    """Per-row share of energy inside the subspace the frame resolves above its noise floor."""
    from entroptics.projection import noise_floor
    A = je.as_frame(H, name="H")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int((s > float(noise_floor(A, far=far, s=s))).sum())
    if k == 0:
        return np.zeros(A.shape[0])
    P = A @ Vt[:k].T
    tot = (A ** 2).sum(1)
    return np.where(tot > 0, (P ** 2).sum(1) / np.where(tot > 0, tot, 1.0), 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--small", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--streams-large", type=Path,
                    default=Path("streams/qwen35_4b_streams.npz"))
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/escalation.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)

    if not a.streams_large.is_file():
        raise SystemExit(f"refusing: {a.streams_large} does not exist")
    z = np.load(a.streams_large)
    big_logits = z["logits"].astype(np.float64)

    tok = transformers.AutoTokenizer.from_pretrained(a.small)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.small).eval()
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    streams, small_logits = collect_streams(m, ids)
    if small_logits.shape != big_logits.shape:
        raise SystemExit(f"refusing: small logits {small_logits.shape} against large "
                         f"{big_logits.shape} -- the two models did not see the same text, so "
                         f"a per-position label would compare different positions")

    agree = (small_logits.argmax(1) == big_logits.argmax(1))
    print(f"{len(agree)} positions, agreement rate {agree.mean():.1%} "
          f"(escalating everything costs 100%, escalating nothing gives that accuracy)")

    e = np.exp(small_logits - small_logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    maxprob = p.max(1)
    negent = (p * np.log(np.clip(p, 1e-30, None))).sum(1)          # negated entropy

    n_layers = streams[0].shape[0]
    rows = []
    print()
    print(f"{'hidden':>6} {'resolved':>10} {'max-softmax':>13} {'negentropy':>12}")
    best = None
    for li in range(1, n_layers):
        res = np.concatenate([resolved_energy_fraction(s[li], far=a.far) for s in streams])
        au = auroc(agree, res)
        rows.append({"hidden_index": li, "auroc_resolved": au})
        if best is None or (not np.isnan(au) and au > best[1]):
            best = (li, au)
    au_mp, au_ne = auroc(agree, maxprob), auroc(agree, negent)
    for r in rows:
        print(f"{r['hidden_index']:>6} {r['auroc_resolved']:>10.4f} "
              f"{au_mp:>13.4f} {au_ne:>12.4f}")
    print()
    print(f"best resolved layer {best[0]} at AUROC {best[1]:.4f}; "
          f"max-softmax {au_mp:.4f}; negentropy {au_ne:.4f}")
    verdict = ("beats" if best[1] > au_mp else "does NOT beat")
    print(f"the screen-based score {verdict} max-softmax")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"small": a.small, "streams_large": str(a.streams_large),
         "positions": int(len(agree)), "agreement_rate": float(agree.mean()),
         "auroc_max_softmax": au_mp, "auroc_negentropy": au_ne,
         "best_layer": int(best[0]), "auroc_resolved_best": float(best[1]),
         "per_layer": rows}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
