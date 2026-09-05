"""Does the linearisation-reach curve hold at a second width?

Paper sec 7.7 measures how far a shared mean-Jacobian transport carries, and reports it from one
model. A depth profile from a single network is a description of that network. Qwen3.5-0.8B is the
second model in the catalogue with a published lens and is small enough to run in float32 locally
(1024 wide against 2560), so the claim is testable rather than merely stated.

Scored in the readout's own terms: the model's logits are `head(rms_norm(x, w))` and a lens's are
`head(rms_norm(J h, w))`, so both sides carry the same normalisation and the comparison is between
the two vectors that are actually unembedded. That also sidesteps the trap in sec 7.7 --
`hidden_states[-1]` arrives already normalised, and it is the *target* for this metric rather than
something needing inversion.

Prompts are drawn exactly as for the 4B run -- wikitext-2-raw-v1 `test`, so out of sample for a
fit taken on train -- at the same count and length, so the two curves are comparable by depth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp4_stream_complement import collect_streams, wikitext_prompts   # noqa: E402


def reach(streams, lens, w):
    """Per-layer centred cosine between the normalised transport and the normalised final state."""
    rows = []
    for l in lens.source_layers:
        J = lens.jacobian(l)
        c = [je.centred_cosine(je.rms_normalize(s[l + 1] @ J.T, w), s[-1]) for s in streams]
        rows.append({"layer": l, "cos": float(np.mean(c)), "cos2": float(np.mean(c)) ** 2,
                     "rel_depth": l / (lens.source_layers[-1] or 1)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--repo", default="models--Qwen--Qwen3.5-0.8B")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"))
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--streams", type=Path, default=Path("streams/qwen35_08b_streams.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/second_model.json"))
    a = ap.parse_args(argv)

    lens = je.load_lens(a.lens)
    w = je.final_norm_weight(a.repo)
    print(f"lens: {len(lens.source_layers)} layers, d={lens.d_model}; "
          f"final norm gain d={w.size}")
    if w.size != lens.d_model:
        raise ValueError(f"norm gain is {w.size} wide, lens is {lens.d_model}")

    if a.streams.exists():
        z = np.load(a.streams)
        S = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                     key=lambda k: int(k[1:]))]
        print(f"reusing {a.streams} ({len(S)} streams)")
    else:
        import torch
        import transformers
        torch.set_grad_enabled(False)
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        m = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
        if next(m.parameters()).dtype != torch.float32:
            m = m.float()                     # bf16 carries ~3 digits; the effect is smaller
        ids = wikitext_prompts(tok, a.prompts, a.tokens)
        S, _ = collect_streams(m, ids)
        S = [s.astype(np.float64) for s in S]
        a.streams.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.streams, **{f"s{i}": s for i, s in enumerate(S)})
        print(f"wrote {a.streams}")
        del m

    rows = reach(S, lens, w)
    print()
    print(f"{'layer':>6}{'rel depth':>11}{'cos':>9}{'cos^2':>9}")
    for r in rows:
        print(f"{r['layer']:>6}{r['rel_depth']:>11.2f}{r['cos']:>9.4f}{r['cos2']:>9.3f}")

    below = [r for r in rows if r["rel_depth"] <= 0.25]
    cross = next((r["rel_depth"] for r in rows if r["cos2"] >= 0.5), None)
    print()
    print(f"mean cos^2 in the first quarter of depth: {np.mean([r['cos2'] for r in below]):.3f}")
    print(f"crosses half the variance at relative depth: "
          f"{'never' if cross is None else format(cross, '.2f')}")
    je.dump(a.out, {"model": a.model, "d_model": lens.d_model,
                    "streams": len(S), "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
