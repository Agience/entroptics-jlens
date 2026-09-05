"""A synthetic lens with a planted workspace band -- the rehearsal rig for experiment 1.

Writes a checkpoint in exactly the format ``jlens.JacobianLens.save`` produces (float16
transports under ``"J"``), with a known resolved rank per layer, so the whole read path can be
exercised end to end before a real ``lens.pt`` is in hand. The planted band is the shape the
paper reports -- a rank in the middle layers, nothing at the ends.

    python experiments/make_synthetic_lens.py --out out/synthetic_lens.pt
    python experiments/exp1_transport_spectrum.py --lens out/synthetic_lens.pt \
        --layers all --controls 3,10,17

Expected: ``K`` equals the planted rank inside the band and 0 outside it.
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
    ap.add_argument("--out", type=Path, default=Path("out/synthetic_lens.pt"))
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--band", default="6,14", help="first,last layer of the planted band")
    ap.add_argument("--rank", type=int, default=10, help="planted rank inside the band")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    lo, hi = (int(v) for v in a.band.split(","))
    rng = np.random.default_rng(a.seed)
    d = a.d_model
    J = {}
    for l in range(a.layers):
        A = rng.standard_normal((d, d))
        if lo <= l <= hi:
            # Amplitudes well above the BBP threshold (~2*sqrt(d)): a spike of strength s is
            # observed at ~s + d/s, so sub-threshold planting is invisible by construction.
            U = je.haar_orthogonal(d, rng)[:, :a.rank]
            V = je.haar_orthogonal(d, rng)[:, :a.rank]
            A = A + (U * np.linspace(90.0, 60.0, a.rank)) @ V.T
        J[l] = torch.tensor(A, dtype=torch.float16)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"J": J, "n_prompts": 1000, "source_layers": list(range(a.layers)),
                "d_model": d}, a.out)
    print(f"wrote {a.out}: d_model={d}, {a.layers} layers, "
          f"planted rank {a.rank} on layers {lo}-{hi}, 0 elsewhere")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, je.IncompleteResults,
            ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
