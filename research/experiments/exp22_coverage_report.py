"""The coverage figure the report leads with, as a script that can be re-run.

`results/coverage_gpt2.json` backs the report's headline -- the real transport covering 13-22% of a
stream's resolved subspace at 23-84x the chance level, where a random rank-K map sits at chance. It
was computed ad hoc and no committed script produced it, so the number could not be reproduced,
let alone re-run on different prompts. That is a worse gap than a missing provenance stamp: the
provenance of an unrepeatable computation is beside the point.

This is that computation, written out. For each layer:

    real        coverage of the stream's resolved subspace by the transported frame
    random      the same through a random map of the same rank -- the control that killed the
                complement read (sec 7.0), applied to the read that replaced it
    null        k_readout / d, the analytic chance level for a readout of that size

and the ratio real/null, which is the "23-84x" of the report. `--skip` draws a second prompt
sample, so the figure can be checked against a draw it was not built on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp4_stream_complement import (collect_streams,                   # noqa: E402
                                    wikitext_prompts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--lens", type=Path, default=Path(
        "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"))
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--skip", type=int, default=0,
                    help="discard this many leading sequences, to draw a second sample")
    ap.add_argument("--residual", action="store_true",
                    help="read M = J - alpha I rather than raw J, as sec 3 requires")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="for the random-map control")
    ap.add_argument("--out", type=Path, default=Path("results/coverage_report.json"))
    a = ap.parse_args(argv)

    import hashlib
    import torch
    import transformers
    torch.set_grad_enabled(False)
    lens = je.load_lens(a.lens)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    m = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    ids = wikitext_prompts(tok, a.prompts, a.tokens, skip=a.skip)
    fp = hashlib.sha256(b"".join(bytes(memoryview(i.numpy())) for i in ids)).hexdigest()[:16]
    streams, _ = collect_streams(m, ids)
    del m
    rng = np.random.default_rng(a.seed)
    print(f"{len(streams)} prompts, fingerprint {fp}")

    rows = []
    print()
    print(f"{'layer':>6}{'real':>9}{'null':>9}{'real/null':>11}{'random':>9}{'rnd/null':>10}")
    for l in lens.source_layers:
        J0 = lens.jacobian(l)
        # The identity core carries no structure and covers everything trivially, so every other
        # read in this work is taken on M = J - alpha I (sec 3). Coverage on raw J climbs to 0.72
        # at gpt2's last layer purely because J -> alpha I with depth; on M it does not.
        J = je.decompose(J0, kind="identity").residual if a.residual else J0
        r_cov, r_null, x_cov, x_null = [], [], [], []
        for s in streams:
            H = s[l + 1]
            c = je.coverage(H, H @ J.T, far=a.far)
            r_cov.append(c.coverage)
            r_null.append(c.null)
            # a random map of the same rank: the control, on the same frame
            k = max(1, c.k_readout)
            Q = np.linalg.qr(rng.standard_normal((H.shape[1], k)))[0]
            cx = je.coverage(H, (H @ Q) @ Q.T, far=a.far)
            x_cov.append(cx.coverage)
            x_null.append(cx.null)
        row = {"layer": l, "real": float(np.mean(r_cov)), "real_null": float(np.mean(r_null)),
               "random": float(np.mean(x_cov)), "random_null": float(np.mean(x_null))}
        row["ratio"] = row["real"] / row["real_null"] if row["real_null"] else float("nan")
        row["random_ratio"] = (row["random"] / row["random_null"]
                               if row["random_null"] else float("nan"))
        rows.append(row)
        print(f"{l:>6}{row['real']:>9.4f}{row['real_null']:>9.4f}{row['ratio']:>11.1f}"
              f"{row['random']:>9.4f}{row['random_ratio']:>10.1f}", flush=True)

    real = [r["real"] for r in rows]
    ratio = [r["ratio"] for r in rows]
    rnd = [r["random_ratio"] for r in rows]
    print()
    print(f"real coverage    {min(real):.1%} - {max(real):.1%}")
    print(f"real / null      {min(ratio):.0f}x - {max(ratio):.0f}x")
    print(f"random / null    {min(rnd):.1f}x - {max(rnd):.1f}x   (chance is 1x)")
    je.dump(a.out, {"model": a.model, "lens": str(a.lens), "prompts": len(streams),
                    "tokens": a.tokens, "prompt_skip": a.skip, "prompt_fingerprint": fp,
                    "far": a.far, "seed": a.seed, "residual": bool(a.residual),
                    "layers": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
