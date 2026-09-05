"""Is the structure training keeps already present before training?

Every efficiency read here failed for one reason: a trained model has already discarded what it
does not need, so asking whether more can be discarded competes with training on its own ground.

That reframing has a consequence worth testing rather than admiring. If training's work is
*discarding*, the structure it keeps may already exist in the geometry beforehand -- training
amplifying what is there rather than inventing it. If so, a derived read could locate that
structure with no gradient descent at all.

The test needs no labels and no fitting:

    run one text through a TRAINED model and through a RANDOMLY INITIALISED model of identical
    architecture, and ask whether the subspaces their activations resolve overlap above chance.

    coverage(random, trained)   how much of the untrained frame's resolved subspace the trained
                                one spans
    null                        k_t / d, what that overlap would be for unrelated subspaces
    control                     a second random init with a different seed: two untrained models
                                share an architecture but no training, so their overlap is the
                                architectural floor rather than evidence of anything learned

Three outcomes, all informative:

    overlap at chance          training builds structure that was not there. The geometry before
                               training carries nothing to find, and "replace training with a
                               derived read" is dead as stated.
    overlap above chance, and  the architecture alone induces the structure. Interesting, but it
    matched by the control     says nothing about what training kept.
    above chance and ABOVE     the trained subspace is present in the untrained geometry beyond
    the control                what architecture explains -- training amplified something a
                               derived read could have located.

The floor applies here: activations on one input are one sample and carry real noise, unlike a
trained weight, whose corpus average leaves no bulk (sec 4).
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompts", type=int, default=32,
                    help="concatenated into one frame: T/d must exceed 1 for a "
                         "floor to read the model rather than the aspect ratio")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/structure_before_training.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    trained = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()
    cfg = transformers.AutoConfig.from_pretrained(a.model)

    def fresh(seed):
        torch.manual_seed(seed)
        return transformers.AutoModelForCausalLM.from_config(cfg).eval().float()

    rand_a, rand_b = fresh(0), fresh(1)
    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    print(f"{a.model}: trained against two random inits, {len(ids)} prompts", flush=True)

    def streams(model):
        out = []
        for s in ids:
            hs = model(s.unsqueeze(0), output_hidden_states=True).hidden_states
            out.append([h[0].numpy().astype(np.float64) for h in hs])
        return out

    S_t, S_a, S_b = streams(trained), streams(rand_a), streams(rand_b)
    L = len(S_t[0]) - 1

    # Concatenate across prompts before reading any floor. One 128-token sequence against d=768 is
    # 128 samples estimating structure in 768 dimensions: the floor then reads the aspect ratio
    # rather than the model, which is what produced a resolved rank of 5-8 where the well-sampled
    # answer was 79. Comparing two frames also requires them at ONE density, or their thresholds
    # mean different things.
    def joined(S, li):
        return np.concatenate([s[li] for s in S])

    d_model = S_t[0][1].shape[1]
    print(f"  concatenated: {joined(S_t, 1).shape[0]} tokens against d={d_model} "
          f"(T/d = {joined(S_t, 1).shape[0] / d_model:.1f})", flush=True)

    rows = []
    print()
    print(f"{'layer':>6}{'K rand':>8}{'K trained':>11}{'rand->trained':>15}"
          f"{'rand->rand':>12}{'null':>9}")
    for li in range(1, L + 1):
        Jt, Ja, Jb = joined(S_t, li), joined(S_a, li), joined(S_b, li)
        c1 = je.coverage(Ja, Jt, far=a.far)                  # untrained signal, trained readout
        c2 = je.coverage(Ja, Jb, far=a.far)                  # untrained against untrained
        row = {"layer": li, "k_random": float(c1.k_signal), "k_trained": float(c1.k_readout),
               "rand_to_trained": float(c1.coverage),
               "rand_to_rand": float(c2.coverage), "null": float(c1.null)}
        rows.append(row)
        print(f"{li:>6}{row['k_random']:>8.1f}{row['k_trained']:>11.1f}"
              f"{row['rand_to_trained']:>15.4f}{row['rand_to_rand']:>12.4f}"
              f"{row['null']:>9.4f}", flush=True)
        je.dump(a.out, {"model": a.model, "prompts": len(ids), "layers": rows}, complete=False)

    rt = np.array([r["rand_to_trained"] for r in rows])
    rr = np.array([r["rand_to_rand"] for r in rows])
    nl = np.array([r["null"] for r in rows])
    print()
    print(f"mean overlap untrained->trained {rt.mean():.4f}  "
          f"({rt.mean() / max(nl.mean(), 1e-12):.1f}x the null)")
    print(f"mean overlap untrained->untrained {rr.mean():.4f}  "
          f"({rr.mean() / max(nl.mean(), 1e-12):.1f}x the null)")
    print()
    if rt.mean() <= 2 * nl.mean():
        print("VERDICT  the untrained geometry carries nothing the trained model uses: "
              "training builds the structure rather than amplifying it")
    elif rt.mean() <= rr.mean() * 1.1:
        print("VERDICT  overlap is explained by the architecture -- two untrained models share "
              "as much as an untrained shares with a trained one")
    else:
        print(f"VERDICT  the trained subspace is present in the untrained geometry beyond what "
              f"architecture explains: {rt.mean() / max(rr.mean(), 1e-12):.2f}x the "
              f"untrained-to-untrained overlap")
    je.dump(a.out, {"model": a.model, "prompts": len(ids), "layers": rows,
                    "mean_rand_to_trained": float(rt.mean()),
                    "mean_rand_to_rand": float(rr.mean()),
                    "mean_null": float(nl.mean())}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
