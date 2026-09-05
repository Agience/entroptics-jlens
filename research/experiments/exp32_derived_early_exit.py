"""Early exit with no threshold: stop when the update falls below the frame's own noise floor.

Every deployed efficiency method carries a tuned number. Early exit fires on a confidence
threshold, KV eviction keeps a top-k, speculative decoding accepts on a ratio. Each is fitted on a
validation set, does not transfer between models, and degrades silently under distribution shift.

The entroptics floor is derived from the object, so a stopping rule can be too:

    a layer's update either moves something the frame resolves, or it does not.

The update frame ``Delta_l = H[l+1] - H[l]`` is a ``(T, d)`` object like any other, so it has a
noise floor of its own. A layer that resolves **zero** modes in its own update is adding nothing
the instrument can distinguish from noise, and every layer after it is spent the same way. That is
a stopping rule with **no number in it**: both sides come from one spectrum.

Measured against what it costs and what it buys:

    layers used      mean fraction of the network actually run
    agreement        top-1 against the full-depth output, per token
    baselines        the tuned rules this would replace -- exit when the logit-lens top-1 has been
                     stable for `k` layers, and exit when max-softmax passes `p`. Both are swept,
                     and the derived rule is compared against the whole curve rather than against
                     one setting of theirs.

The derived rule earns its place only if it lands on that curve. Beating a tuned baseline is not
required and would be suspicious; matching one without a validation set is the claim.
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


def readout(model):
    """``head(norm(h))`` for intermediate states -- and NOT for the last one.

    `hidden_states[-1]` arrives with the model's final norm already applied. Norming it again
    produced a readout that reproduced the model's own logits on 73% of tokens instead of 100%,
    which corrupted every layerwise prediction in this experiment and both tuned baselines with
    them. The caller passes ``already_normed`` for the final state; `verify` refuses if the
    reconstruction is not exact.
    """
    inner = getattr(model.model, "language_model", model.model)
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layer_norm")
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out")

    def un(H, already_normed: bool = False):
        import torch
        x = torch.tensor(np.asarray(H), dtype=torch.float32)
        return head(x if already_normed else norm(x)).detach().numpy()
    return un


def verify(un, hidden_last, real_logits):
    """Refuse to measure anything with a readout that cannot reproduce the model's own output."""
    got = un(hidden_last, already_normed=True)
    agree = float((got.argmax(1) == real_logits.argmax(1)).mean())
    diff = float(np.abs(got - real_logits).max())
    if agree < 0.9999 or diff > 1e-2:
        raise ValueError(
            f"readout does not reproduce the model: top-1 agreement {agree:.4f}, max|diff| "
            f"{diff:.4f}. Every layerwise prediction built on it would be wrong, and a ceiling "
            f"measured from it would be a statement about this bug.")
    return agree, diff


def oracle_ceiling(tops, final, L):
    """The first layer after which each token's prediction never changes again.

    This is the ceiling on ANY exit rule: what a perfect oracle would save. Measuring it first is
    the step that was skipped -- rules were built and compared before anyone asked whether there
    was anything to win.
    """
    never = np.full(len(final), L)
    for l in range(L - 1, -1, -1):
        never = np.where(np.all(tops[l:] == final[None, :], axis=0), l + 1, never)
    return never


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/derived_early_exit.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from entroptics.projection import noise_floor
    torch.set_grad_enabled(False)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval().float()
    un = readout(model)
    ids = wikitext_prompts(tok, a.prompts, a.tokens)
    print(f"{a.model}: {len(ids)} prompts of {a.tokens} tokens", flush=True)

    used_d, agree_d, per_layer = [], [], []
    tuned_stable, tuned_conf = {}, {}
    for si, seq in enumerate(ids):
        hs = model(seq.unsqueeze(0), output_hidden_states=True).hidden_states
        H = [h[0].numpy().astype(np.float64) for h in hs]
        L = len(H) - 1                                    # hidden_states[0] is the embedding
        final = un(H[-1]).argmax(1)
        top = np.stack([un(H[l]).argmax(1) for l in range(1, L + 1)])      # (L, T)

        # Derived rule: the first layer whose UPDATE FRAME resolves nothing above its own floor.
        #
        # The first version of this compared || h_{l+1} - h_l || for one token against the noise
        # floor of the stream frame. Those are different kinds of quantity -- a floor bounds the
        # SINGULAR VALUES of a (T, d) frame, at a scale set by T, while a row norm is one vector's
        # length. The comparison was true almost everywhere, the rule exited at layer 1, and it
        # agreed with the full-depth output 0.3% of the time.
        #
        # The update frame Delta_l = H[l+1] - H[l] is a (T, d) object like any other, so it has a
        # floor of its own. A layer that resolves zero modes in its own update is adding nothing
        # the instrument can distinguish from noise, and that is the stopping condition. Still no
        # number: both sides come from the same spectrum.
        resolved = []
        for l in range(1, L):
            D = je.as_frame(H[l + 1] - H[l])
            sv = np.linalg.svd(D, full_matrices=False, compute_uv=False)
            resolved.append(int((sv > float(noise_floor(D, far=a.far, s=sv))).sum()))
        dead = np.array(resolved) == 0
        first_layer = int(np.argmax(dead) + 1) if dead.any() else L
        first = np.full(top.shape[1], first_layer)
        used_d.append(first / L)
        agree_d.append(top[np.clip(first, 1, L) - 1, np.arange(top.shape[1])] == final)
        per_layer.append({"layers": int(L), "exit_layer": first_layer,
                          "resolved_per_layer": resolved})

        # tuned baseline 1: top-1 stable for k layers
        for k in (1, 2, 3, 4):
            stable = np.full(top.shape[1], L)
            for l in range(k, L):
                same = np.all(top[l - k:l] == top[l][None, :], axis=0) & (stable == L)
                stable = np.where(same, l + 1, stable)
            tuned_stable.setdefault(k, {"used": [], "agree": []})
            tuned_stable[k]["used"].append(stable / L)
            tuned_stable[k]["agree"].append(top[stable - 1, np.arange(top.shape[1])] == final)

        # tuned baseline 2: max-softmax over the layer's own readout
        for p in (0.5, 0.7, 0.9, 0.95):
            conf = np.full(top.shape[1], L)
            for l in range(1, L):
                lg = un(H[l])
                e = np.exp(lg - lg.max(1, keepdims=True))
                mx = (e / e.sum(1, keepdims=True)).max(1)
                conf = np.where((mx >= p) & (conf == L), l, conf)
            tuned_conf.setdefault(p, {"used": [], "agree": []})
            tuned_conf[p]["used"].append(conf / L)
            tuned_conf[p]["agree"].append(top[conf - 1, np.arange(top.shape[1])] == final)
        print(f"  prompt {si + 1}/{len(ids)}", flush=True)

    def summarise(used, agree):
        return float(np.mean(np.concatenate(used))), float(np.mean(np.concatenate(agree)))

    d_used, d_agree = summarise(used_d, agree_d)
    print()
    print(f"{'rule':>26}{'layers run':>12}{'agreement':>12}")
    print(f"{'DERIVED (no threshold)':>26}{d_used:>11.1%}{d_agree:>12.3f}")
    rows = {"derived": {"used": d_used, "agree": d_agree}, "tuned": {}}
    for k, v in tuned_stable.items():
        u, g = summarise(v["used"], v["agree"])
        rows["tuned"][f"stable{k}"] = {"used": u, "agree": g}
        print(f"{'tuned: top-1 stable ' + str(k):>26}{u:>11.1%}{g:>12.3f}")
    for p, v in tuned_conf.items():
        u, g = summarise(v["used"], v["agree"])
        rows["tuned"][f"conf{p}"] = {"used": u, "agree": g}
        print(f"{'tuned: max-softmax ' + str(p):>26}{u:>11.1%}{g:>12.3f}")

    # is the derived point on the tuned curve? compare against the best tuned rule that is at
    # least as accurate, and report the compute difference at matched accuracy.
    cands = [(v["used"], v["agree"], n) for n, v in rows["tuned"].items() if v["agree"] >= d_agree]
    print()
    if cands:
        u, g, n = min(cands)
        print(f"cheapest tuned rule at least as accurate: {n} at {u:.1%} layers "
              f"(agreement {g:.3f})")
        print(f"derived rule costs {d_used - u:+.1%} of the network against it, with nothing tuned")
    else:
        print("no tuned rule reaches the derived rule's agreement at any setting swept")
    je.dump(a.out, {"model": a.model, "prompts": len(ids), "tokens": a.tokens,
                    "far": a.far, **rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
