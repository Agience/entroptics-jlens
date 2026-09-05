"""P2 -- selective prediction against ground truth.

P1 asked whether a small model agrees with a large one. That label lives in output space, which
advantages a confidence score read off the logits and disadvantages a geometric one -- a fair
criticism of that benchmark. This one removes the objection: the label is **ground truth**.

    label   at each position, did the model predict the ACTUAL next token? Wikitext supplies the
            answer, so no second model is involved and no model's opinion stands in for correct.
    scores  from the model alone.
    metrics AUROC, and AURC -- area under the risk-coverage curve, which is what selective
            prediction is actually judged on: as you abstain on more positions, how fast does
            error on the ones you keep fall?

Two questions, and the second is the one that matters once the first is lost:

    standalone    does a geometric score beat max-softmax on its own? (P1 says no; expect no.)
    incremental   does it add anything *on top of* max-softmax? A feature that carries
                  information the baseline lacks is worth having even if it loses alone. Tested
                  by cross-validated logistic regression on (maxprob) against
                  (maxprob, geometric), which is the standard way to ask.

Needs only the small model, so it runs on many more positions than P1 could.
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
from exp7_escalation import auroc, resolved_energy_fraction            # noqa: E402


def aurc(y_correct, score):
    """Area under the risk-coverage curve. Lower is better; 1 - accuracy is the no-skill value.

    Sort by confidence descending, keep the top `c` fraction, and record the error rate among
    those kept. A score that ranks errors last drives risk down as coverage falls.
    """
    y = np.asarray(y_correct).astype(bool)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    err = (~y[order]).astype(np.float64)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def logistic_cv_auroc(X, y, folds=5, seed=0, iters=300, lr=0.5):
    """Cross-validated AUROC of a logistic model. Plain gradient descent -- one statistic does
    not justify a scikit-learn dependency, and standardising inside each fold keeps the test
    fold out of the fit."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    out = np.empty(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        B = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd])
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            g = A.T @ (1.0 / (1.0 + np.exp(-A @ w)) - y[tr]) / len(tr)
            w -= lr * g
        out[te] = B @ w
    return auroc(y.astype(bool), out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/selective.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    streams, logits = collect_streams(m, ids)

    # position t predicts token t+1, so the last position of each prompt has no label
    keep, targets = [], []
    off = 0
    for s in ids:
        arr = s.numpy()
        for t in range(len(arr) - 1):
            keep.append(off + t)
            targets.append(arr[t + 1])
        off += len(arr)
    keep = np.array(keep)
    targets = np.array(targets)
    L = logits[keep]
    correct = (L.argmax(1) == targets)
    print(f"{len(correct)} labelled positions, next-token accuracy {correct.mean():.1%}")

    e = np.exp(L - L.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    maxprob = p.max(1)
    negent = (p * np.log(np.clip(p, 1e-30, None))).sum(1)
    margin = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]

    n_hidden = streams[0].shape[0]
    res_all = {}
    for li in range(1, n_hidden):
        r = np.concatenate([resolved_energy_fraction(s[li], far=a.far) for s in streams])
        res_all[li] = r[keep]

    base = {"max-softmax": maxprob, "negentropy": negent, "margin": margin}
    print()
    print(f"{'score':<26}{'AUROC':>9}{'AURC':>9}")
    rows = []
    for name, sc in base.items():
        rows.append({"score": name, "auroc": auroc(correct, sc), "aurc": aurc(correct, sc)})
        print(f"{name:<26}{rows[-1]['auroc']:>9.4f}{rows[-1]['aurc']:>9.4f}")
    best_li = max(res_all, key=lambda k: auroc(correct, res_all[k]))
    rows.append({"score": f"resolved (hidden {best_li})",
                 "auroc": auroc(correct, res_all[best_li]),
                 "aurc": aurc(correct, res_all[best_li])})
    print(f"{rows[-1]['score']:<26}{rows[-1]['auroc']:>9.4f}{rows[-1]['aurc']:>9.4f}")
    print(f"{'no-skill (1 - accuracy)':<26}{0.5:>9.4f}{1 - correct.mean():>9.4f}")

    print()
    print("incremental value over max-softmax, 5-fold cross-validated logistic:")
    a_base = logistic_cv_auroc(maxprob, correct)
    print(f"  {'maxprob alone':<34}{a_base:>9.4f}")
    gains = []
    for li in sorted(res_all):
        a_both = logistic_cv_auroc(np.column_stack([maxprob, res_all[li]]), correct)
        gains.append({"hidden": li, "auroc": a_both, "gain": a_both - a_base})
    top = sorted(gains, key=lambda g: -g["gain"])[:3]
    for g in top:
        print(f"  {'+ resolved hidden ' + str(g['hidden']):<34}{g['auroc']:>9.4f}"
              f"   gain {g['gain']:+.4f}")
    best_gain = top[0]["gain"]
    print()
    print(f"verdict: best incremental gain {best_gain:+.4f} AUROC over max-softmax "
          f"({'worth having' if best_gain > 0.01 else 'not material'})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"model": a.model, "positions": int(len(correct)),
         "accuracy": float(correct.mean()), "standalone": rows,
         "logistic_base_auroc": a_base, "incremental": gains}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
