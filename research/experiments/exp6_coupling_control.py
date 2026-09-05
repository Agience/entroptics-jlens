"""Is cross-model coupling layer-specific, or just shared token statistics?

``coupling``'s permutation null shuffles rows, so it tests position alignment. It does not test
whether the LEVEL of coupling is explained by both models knowing the same unigram structure --
two logit frames over one vocabulary share that whatever layer they came from.

Two controls:
  mismatched layer   couple A's layer against B's wrong layer. If that couples as strongly as
                     the matched pair, the number is about the vocabulary, not the layers.
  column-centred     subtract each side's per-token mean before coupling, which removes the
                     shared frequency structure and leaves position-specific agreement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from entroptics import Screen                                          # noqa: E402
from exp4_stream_complement import (collect_streams, unembed_fn,       # noqa: E402
                                    wikitext_prompts)
from exp5_cross_model import load_readout, rms_unembed                 # noqa: E402


def couple(Xa, Xb, centre: bool):
    if centre:
        Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)
    s = Screen(far=0.05)
    s.register("a", entry=lambda X: np.asarray(X, dtype=np.float64))
    s.register("b", entry=lambda X: np.asarray(X, dtype=np.float64))
    s.place("a", Xa)
    s.place("b", Xb)
    c = s.coupling("a", "b")
    return float(c.strength), float(c.z)


def main(argv=None) -> int:
    # A parser before anything else, and that ordering is the point rather than a convention.
    # Without one, `--help` was ignored and this function ran -- which here means downloading a
    # model and writing results/. The same shape once had a `--help` re-run an experiment that a
    # timeout then killed at layer 26 of 31, leaving a truncated file a later script read as
    # whole (see `entroptics_jlens.results`).
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens-a", type=Path, default=Path(
        "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"))
    ap.add_argument("--lens-b", type=Path, default=Path(
        "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"))
    ap.add_argument("--streams", type=Path, default=Path("streams/qwen35_4b_streams.npz"))
    ap.add_argument("--model-a", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--model-b", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("results/coupling_control.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)
    la = je.load_lens(a.lens_a)
    lb = je.load_lens(a.lens_b)
    z = np.load(a.streams)
    Sb = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                  key=lambda k: int(k[1:]))]
    lg_b = z["logits"].astype(np.float64)
    tok = transformers.AutoTokenizer.from_pretrained(a.model_a)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.model_a).eval().float()
    Sa, lg_a = collect_streams(m, wikitext_prompts(tok, a.prompts, a.tokens))

    def mass(L):
        e = np.exp(L - L.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).sum(0)
    score = 0.5 * (mass(lg_a) / len(lg_a) + mass(lg_b) / len(lg_b))
    keep = np.sort(np.argpartition(-score, 4096)[:4096])
    un_a, (nb, hb) = unembed_fn(m), load_readout(a.model_b)
    un_b = rms_unembed(nb, hb, keep)

    fa = {l: np.concatenate([un_a(s[l + 1] @ la.jacobian(l).T)[:, keep] for s in Sa], 0)
          for l in (5, 11, 18)}
    fb = {l: np.concatenate([un_b(s[l + 1] @ lb.jacobian(l).T) for s in Sb], 0)
          for l in (0, 2, 7, 15, 25, 30)}

    rows = []
    print(f"{'A':>4}{'B':>4}  {'pairing':<12}{'raw':>9}{'z':>8}{'centred':>10}{'z':>8}",
          flush=True)
    for a_l, matched, wrong in ((5, 7, 30), (11, 15, 0), (18, 25, 2)):
        for tag, b_l in (("matched", matched), ("mismatched", wrong)):
            r, rz = couple(fa[a_l], fb[b_l], False)
            c, cz = couple(fa[a_l], fb[b_l], True)
            rows.append({"layer_a": a_l, "layer_b": b_l, "pairing": tag,
                         "raw": r, "raw_z": rz, "centred": c, "centred_z": cz})
            print(f"{a_l:>4}{b_l:>4}  {tag:<12}{r:>9.4f}{rz:>8.1f}{c:>10.4f}{cz:>8.1f}",
                  flush=True)
            # Through `je.dump`, not a bare write. This loop writes on every iteration so a
            # killed run keeps what it computed -- which is exactly the shape that once left a
            # truncated file a later script read as a finished sweep. `complete=False` here and
            # `complete=True` after the loop is what makes the difference visible to a reader.
            je.dump(a.out, {"keep": int(keep.size), "rows": rows}, complete=False)
    je.dump(a.out, {"keep": int(keep.size), "rows": rows}, complete=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults,
            ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
