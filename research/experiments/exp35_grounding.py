"""Can a geometric read catch what the model's own probability gets confidently wrong?

Six efficiency attempts failed here and the reason is structural: a trained model has already
discarded everything it does not need, so asking whether more can be discarded is competing with
training on its own ground. The compression ceilings being flat is training working, not the
instrument failing.

**And a trained weight has no noise floor at all.** sec 4 measures it: two independently fitted
Jacobians of one model correlate 0.9999, the corpus average removed the incoherent part, and
Tracy-Widom has no bulk whose edge it could locate. A floor derived from a spectrum needs a
spectrum with noise in it. So the floor does not apply to weights or to averaged transports -- it
applies to **one sample of activations on one input**, which is a different object and does carry
noise. That partition decides where this instrument is eligible to say anything, and every read
below is on the eligible side.

The 1% that matters is not efficiency. It is the model being confident and wrong.

    max-softmax is the model grading its own answer, so it is *by construction* uninformative
    exactly where the model is confident and wrong.

A geometric read of the activation frame is not derived from the model's probability, and
independence is the whole requirement for catching confident errors. The concrete form: a
hallucination is an answer the context does not support, which is a **coverage** question --

    grounding = coverage of the question's resolved subspace by the answer's frame, against the
                analytic null for a readout of that size

TruthfulQA is the right test because it is adversarial to the baseline by construction: its false
answers are common misconceptions, chosen to be MORE probable under a language model than the true
ones. A model's own log-probability is expected to do badly. If a geometric read does better, the
signal is real and it is independent.

    baseline    the model's length-normalised log-probability of each choice
    grounding   coverage excess of the question's frame by the choice's frame
    combined    both, to see whether the geometric read adds anything the baseline lacks
    truth       TruthfulQA mc1 labels: one true answer among plausible falsehoods, 817 questions,
                human-written, and nothing here influenced them
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--layer", type=float, default=0.6,
                    help="relative depth to read the frame at; the reach curve (sec 7.7) says a "
                         "shared linearisation is meaningless early, so mid-to-late is where a "
                         "frame carries the answer's structure")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/grounding.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from datasets import load_dataset
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rows_in = [r for r in ds][: a.questions]
    print(f"{a.model} on {len(rows_in)} TruthfulQA mc1 questions", flush=True)

    def frames(text):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=192)
        out = model(**ids, output_hidden_states=True)
        n = len(out.hidden_states) - 1
        li = max(1, min(n, int(round(a.layer * n))))
        return (out.hidden_states[li][0].numpy().astype(np.float64),
                out.logits[0].double(), ids["input_ids"][0])

    def logprob(q, ans):
        """Length-normalised log-probability of the answer given the question."""
        qi = tok(q, return_tensors="pt").input_ids
        full = tok(q + " " + ans, return_tensors="pt", truncation=True, max_length=192).input_ids
        if full.shape[1] <= qi.shape[1]:
            return -1e9
        lg = model(full).logits[0].double()
        lp = torch.log_softmax(lg, -1)
        tgt = full[0, qi.shape[1]:]
        got = lp[qi.shape[1] - 1: -1].gather(1, tgt[:, None]).squeeze(1)
        return float(got.mean())

    rows = []
    for i, r in enumerate(rows_in):
        q = r["question"]
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        qf, _, _ = frames(q)
        lps, cov = [], []
        for ch in choices:
            lps.append(logprob(q, ch))
            cf, _, _ = frames(q + " " + ch)
            cov.append(je.coverage(qf, cf, far=a.far).excess)
        rows.append({"q": i, "labels": [int(v) for v in labels],
                     "logprob": lps, "grounding": cov})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows_in)}", flush=True)

    def accuracy(key, sign=1.0):
        hit = [int(np.argmax(sign * np.asarray(r[key])) == int(np.argmax(r["labels"])))
               for r in rows]
        return float(np.mean(hit)), np.array(hit)

    acc_lp, hit_lp = accuracy("logprob")
    acc_gr, hit_gr = accuracy("grounding")
    # combined: rank-average the two, which needs no fitted weight
    comb = []
    for r in rows:
        rl = np.argsort(np.argsort(r["logprob"]))
        rg = np.argsort(np.argsort(r["grounding"]))
        comb.append(int(np.argmax(rl + rg) == int(np.argmax(r["labels"]))))
    acc_cb = float(np.mean(comb))
    chance = float(np.mean([1.0 / len(r["labels"]) for r in rows]))

    print()
    print(f"{'signal':>26}{'accuracy':>10}")
    print(f"{'chance':>26}{chance:>10.3f}")
    print(f"{'model log-probability':>26}{acc_lp:>10.3f}")
    print(f"{'grounding (geometric)':>26}{acc_gr:>10.3f}")
    print(f"{'rank-average of both':>26}{acc_cb:>10.3f}")

    rng = np.random.default_rng(0)
    # The combination is the claim, so it gets its own interval. The rule is a fixed rank-average:
    # no weight is fitted, so there is nothing here tuned on the answers.
    hit_cb = np.array(comb, dtype=float)
    dc = hit_cb - hit_lp.astype(float)
    bsc = dc[rng.integers(0, len(dc), size=(4000, len(dc)))].mean(1)
    lo_c, hi_c = np.percentile(bsc, [2.5, 97.5])
    print()
    print(f"combined - logprob {dc.mean():+.4f}  95% CI [{lo_c:+.4f}, {hi_c:+.4f}]  "
          f"{'significant' if lo_c > 0 or hi_c < 0 else 'not significant'}")
    # Independence is the mechanism: two signals at the same accuracy that fail in different
    # places are worth more together than either is alone. Expected overlap under independence is
    # the product of the accuracies; less than that means the successes are anti-correlated.
    exp_both = acc_lp * acc_gr
    d = hit_gr.astype(float) - hit_lp.astype(float)
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print()
    print(f"grounding - logprob {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")
    # independence is the property that matters even when accuracy is not higher
    both = float(np.mean(hit_gr & hit_lp))
    either = float(np.mean(hit_gr | hit_lp))
    print(f"both right {both:.3f} (independence predicts {exp_both:.3f})   "
          f"either right {either:.3f}")
    print(f"grounding right where logprob wrong {np.mean(hit_gr & ~hit_lp):.3f}   "
          f"logprob right where grounding wrong {np.mean(hit_lp & ~hit_gr):.3f}")
    je.dump(a.out, {"model": a.model, "questions": len(rows), "chance": chance,
                    "acc_logprob": acc_lp, "acc_grounding": acc_gr, "acc_combined": acc_cb,
                    "delta": {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi)},
                    "both": both, "either": either, "expected_both": float(exp_both),
                    "combined_delta": {"mean": float(dc.mean()), "lo": float(lo_c),
                                       "hi": float(hi_c)},
                    "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
