"""B5 -- does this pipeline reproduce a published ordering it played no part in establishing?

Every other benchmark here is self-consistency: the instrument against its own nulls and its own
constructions. This one checks it against a result obtained by other people, by other means.

[Anthropic 2026] report that the Jacobian lens recovers structure at early layers where the
logit lens fails, because the logit lens assumes representational consistency across depth and
the Jacobian transport corrects for the drift. That is an ordering this work did not produce and
cannot influence.

    logit lens      unembed(h_l)                 -- the residual read directly
    Jacobian lens   unembed(h_l @ J_l^T)         -- transported first

Both are scored the same way, against the model's own final logits: centred cosine over the
vocabulary, and top-1 agreement at positions where the model is confident. If the ordering
reproduces -- Jacobian ahead, and by more at early layers -- the pipeline is measuring what it
claims to. If it does not, something upstream is wrong and every other number here is suspect.

A failure to reproduce would be more informative than the successes elsewhere in this repo.

RESULT: not reproduced -- the logit lens scores ahead at every layer. And the benchmark is the
thing at fault, not the pipeline.

The metric here scores agreement with the model's FINAL output. That is exactly the property
[Anthropic 2026] criticises the tuned lens for: "skipping ahead to outputs rather than surfacing
unverbalized intermediate computations". The Jacobian lens's claimed advantage is surfacing
content that is ABSENT from the output -- their worked example reads "nose" at mid layers where
the word never appears in the prompt at all. A metric rewarding output agreement rewards the
failure mode the paper names, so scoring the J-lens badly on it is what should happen.

This benchmark therefore cannot settle the published ordering, and is kept as a record of why.
The claim is evidenced causally in the original work -- swap a vector, watch the output change --
and nothing in this repo intervenes, so the ordering is not checkable here with the reads
available. That is a limitation of this work, not a finding about the lens.

The same statistic remains valid where it is used elsewhere: `check_alignment` compares OFFSETS
of one lens against each other, which is a relative comparison of the same quantity, and top-1
agreement corroborates it. It is not a measure of lens quality and is not used as one.
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
from exp4_stream_complement import (collect_streams, unembed_fn,       # noqa: E402
                                    wikitext_prompts)


def score(L, model_logits, conf):
    """Centred cosine over the vocabulary, and top-1 agreement where the model is decided."""
    a = L - L.mean(1, keepdims=True)
    b = model_logits - model_logits.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    cos = float(np.mean((a * b).sum(1) / np.maximum(den, 1e-30)))
    top = float((L.argmax(1)[conf] == model_logits.argmax(1)[conf]).mean())
    return cos, top


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/lens_ranking.json"))
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
    streams, model_logits = collect_streams(m, ids)
    un = unembed_fn(m)

    e = np.exp(model_logits - model_logits.max(1, keepdims=True))
    conf = (e / e.sum(1, keepdims=True)).max(1) >= 0.5
    print(f"{len(model_logits)} positions, {int(conf.sum())} confident (p>=0.5)")

    n = lens.source_layers[-1] or 1
    rows = []
    print()
    print(f"{'layer':>6}{'rel':>6}{'logit cos':>11}{'jlens cos':>11}{'gain':>8}"
          f"{'logit top1':>12}{'jlens top1':>12}")
    for l in lens.source_layers:
        H = np.concatenate([s[l + a.offset] for s in streams])
        J = lens.jacobian(l)
        c_lg, t_lg = score(un(H), model_logits, conf)                  # logit lens
        c_jl, t_jl = score(un(H @ J.T), model_logits, conf)            # Jacobian lens
        rows.append({"layer": l, "rel": l / n, "logit_cos": c_lg, "jlens_cos": c_jl,
                     "logit_top1": t_lg, "jlens_top1": t_jl})
        print(f"{l:>6}{l/n:>6.2f}{c_lg:>11.4f}{c_jl:>11.4f}{c_jl - c_lg:>+8.4f}"
              f"{t_lg:>12.3f}{t_jl:>12.3f}", flush=True)

    half = max(1, len(rows) // 2)
    early = rows[:half]
    late = rows[half:]
    ge = float(np.mean([r["jlens_cos"] - r["logit_cos"] for r in early]))
    gl = float(np.mean([r["jlens_cos"] - r["logit_cos"] for r in late]))
    wins = sum(1 for r in rows if r["jlens_cos"] > r["logit_cos"])
    print()
    print(f"Jacobian ahead on {wins}/{len(rows)} layers")
    print(f"mean gain, early half {ge:+.4f}   late half {gl:+.4f}")
    ok = wins > len(rows) / 2 and ge > gl
    print(f"published ordering (Jacobian ahead, and by more early): "
          f"{'REPRODUCED' if ok else 'NOT reproduced'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "positions": int(len(model_logits)),
                                 "confident": int(conf.sum()), "wins": wins,
                                 "gain_early": ge, "gain_late": gl,
                                 "reproduced": bool(ok), "layers": rows}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
