"""P3 -- can an unlabelled read tell you where to attach a probe?

P1 and P2 failed because they asked a frame-level instrument to score individual predictions.
Coverage, etendue and participation ratio all measure a *map*, not a token. So the benchmark
should ask a map-level question, and this is the one with real cost attached:

    A probe is attached to a model at some layer. Which layer should it be? Today that is
    answered by training a probe at every candidate layer and evaluating each on labelled data.
    If an UNLABELLED read predicts the ranking, the labels and most of the training are
    unnecessary -- and the same read answers "does my probe still work after the model upgrade"
    without re-collecting a labelled set.

Setup:

    truth     train a linear probe at each layer on a real task and measure held-out accuracy.
              That ranking is the ground truth this is trying to predict.
    candidate unlabelled reads of the same layers: resolved dimension, participation ratio,
              Shannon rank, and the coverage of the layer's stream by the model's own transport.
    metric    Spearman correlation between each unlabelled read and the probe-accuracy ranking,
              plus whether picking the argmax read lands on a good layer.

The task is part-of-speech-ish by construction and needs no annotation: predict whether the NEXT
token begins with a space, which separates word-initial from word-continuation tokens. It is
linearly decodable to a useful degree, varies across depth, and requires no labelled corpus --
which keeps the benchmark honest about what it is testing.
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


def _midranks(a) -> "np.ndarray":
    """Ranks with ties averaged.

    `argsort(argsort(x))` breaks a tie by position in the array, so the statistic depends on row
    order. Three of the nine models profiled here share d_model = 2560, where the tie-unaware
    estimator reports +0.483 against a mid-rank +0.390.
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _midranks(x), _midranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def probe_accuracy(X, y, folds=4, seed=0, iters=400, lr=0.5):
    """Held-out accuracy of a ridge-regularised linear probe, cross-validated."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    pred = np.empty(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        B = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd])
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            g = A.T @ (1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30))) - y[tr]) / len(tr)
            g[1:] += 1e-3 * w[1:]
            w -= lr * g
        pred[te] = B @ w
    return float(((pred > 0) == (y > 0.5)).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"))
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--target", default="current", choices=("current", "next"),
                    help="which token's property to probe for. 'next' is next-token prediction "
                         "in disguise and is monotonically best at the final layer, which makes "
                         "'use the deepest layer' unbeatable and leaves an unlabelled read "
                         "nothing to find. 'current' peaks in the interior, where the question "
                         "is real.")
    ap.add_argument("--dims", type=int, default=256,
                    help="random projection width for the probe, to keep folds well-posed")
    ap.add_argument("--out", type=Path, default=Path("results/probe_placement.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    lens = je.load_lens(a.lens)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    streams, _ = collect_streams(m, ids)

    # label: does the next token start a new word? No annotation needed.
    lab, sel, off = [], [], 0
    for s in ids:
        arr = s.numpy()
        for t in range(len(arr) - 1):
            j = t if a.target == "current" else t + 1
            tokstr = tok.convert_ids_to_tokens(int(arr[j]))
            lab.append(1.0 if (tokstr.startswith(chr(0x0120))
                               or tokstr.startswith(" ")) else 0.0)
            sel.append(off + t)
        off += len(arr)
    y = np.array(lab)
    sel = np.array(sel)
    print(f"{len(y)} positions, positive rate {y.mean():.1%}, target={a.target!r}")

    d = streams[0].shape[-1]
    rng = np.random.default_rng(0)
    R = rng.standard_normal((d, a.dims)) / np.sqrt(d)     # one projection, shared by all layers

    n_hidden = streams[0].shape[0]
    rows = []
    print()
    print(f"{'hidden':>7}{'probe acc':>11}{'coverage':>10}{'PR':>9}{'H2':>9}{'k_res':>7}")
    for li in range(1, n_hidden):
        H = np.concatenate([s[li] for s in streams])[sel]
        acc = probe_accuracy(H @ R, y)
        Hs = np.concatenate([s[li] for s in streams])
        J = lens.jacobian(li - 1) if (li - 1) in lens.source_layers else None
        cov = je.coverage(Hs, Hs @ J.T).coverage if J is not None else float("nan")
        sp = je.energy_spectrum(Hs)
        rows.append({"hidden": li, "probe_acc": acc, "coverage": cov,
                     "pr": je.participation_ratio(sp), "h2": je.shannon_rank(sp),
                     "k_res": int(je.coverage(Hs, Hs).k_signal)})
        r = rows[-1]
        print(f"{li:>7}{acc:>11.4f}{r['coverage']:>10.4f}{r['pr']:>9.1f}{r['h2']:>9.1f}"
              f"{r['k_res']:>7}", flush=True)

    acc = [r["probe_acc"] for r in rows]
    print()
    print("Spearman against probe accuracy across layers:")
    corr = {}
    for key, lab_ in (("coverage", "coverage"), ("pr", "participation ratio"),
                      ("h2", "Shannon rank"), ("k_res", "resolved dimension")):
        v = [r[key] for r in rows]
        ok = ~np.isnan(np.asarray(v, dtype=float))
        corr[key] = spearman(np.asarray(v, float)[ok], np.asarray(acc)[ok])
        print(f"  {lab_:<22}{corr[key]:>+8.3f}")
    best_true = max(rows, key=lambda r: r["probe_acc"])
    last = rows[-1]
    print(f"  trivial 'deepest layer'        -> hidden {last['hidden']:>2} "
          f"(acc {last['probe_acc']:.4f}); regret "
          f"{best_true['probe_acc'] - last['probe_acc']:+.4f}")
    for key in ("coverage", "pr", "h2"):
        cand = max((r for r in rows if not np.isnan(r[key])), key=lambda r: r[key])
        print(f"  picking argmax {key:<10} -> hidden {cand['hidden']:>2} "
              f"(acc {cand['probe_acc']:.4f}); best is hidden {best_true['hidden']} "
              f"(acc {best_true['probe_acc']:.4f}); "
              f"regret {best_true['probe_acc'] - cand['probe_acc']:+.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "positions": int(len(y)),
                                 "positive_rate": float(y.mean()), "spearman": corr,
                                 "layers": rows}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
