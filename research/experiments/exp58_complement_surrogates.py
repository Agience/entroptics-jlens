"""The complement read of sections 6.1 and 6.2, set beside the matched-spectrum surrogate.

``exp4_stream_complement.py`` writes two kinds of run on the same prompts: the real transport
(``results/complement_*.json``) and, under ``--surrogate SEED``, the same singular values on
Haar-random frames (``results/surrogate_*_s<SEED>.json``). The surrogate shares no directions with
the stream by construction, so the modes its complement removes are what the read removes from
the spectrum alone. What the real transport removes beyond that is the part a reader can attribute
to the transport's frame.

The surrogate is a draw, so its removal is a draw too. Where several seeds were run, the table
gives each seed's removal and the real removal's excess over the largest of them; where one seed
was run, it gives that seed's removal and no spread is available. Nothing is fitted and no
threshold is applied: every number is a difference of two counts already in the result files.

    python research/experiments/exp58_complement_surrogates.py

Prints the six-model summary of section 6.2 (resolved dimension, outside%, J-space energy share,
signed and absolute removal, the seed-0 surrogate's removal), the width trend, and the per-layer
tables of section 6.1 for the two models run at three seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"

#: (label, width, real-run file, surrogate tag). The real file is the eight-prompt run at the
#: participation-ratio rank, six prompts on Qwen3.5-4B; the surrogate runs reuse its prompts.
MODELS = [
    ("pythia-70m", 512, "complement_pythia70m_p8.json", "pythia70m"),
    ("gpt2", 768, "complement_gpt2_p8.json", "gpt2"),
    ("Qwen3.5-0.8B", 1024, "complement_qwen0.8b_p8.json", "qwen0.8b"),
    ("gemma-3-1b", 1152, "complement_gemma31b_all_p8.json", "gemma31b"),
    ("Qwen3-1.7B", 2048, "complement_qwen1.7b_p8.json", "qwen1.7b"),
    ("Qwen3.5-4B", 2560, "complement_qwen35_4b_p6.json", "qwen35_4b"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _layers(run: dict) -> dict:
    return {x["layer"]: x for x in run["layers"]}


def removed(rec: dict) -> float:
    return rec["K_stream"] - rec["K_complement"]


def surrogates(tag: str, fingerprint: str) -> list[tuple[int, dict]]:
    out = []
    for p in sorted(RESULTS.glob(f"surrogate_{tag}_s*.json"),
                    key=lambda q: int(q.stem.rsplit("_s", 1)[1])):
        run = _load(p)
        if run["prompt_fingerprint"] != fingerprint:
            raise SystemExit(f"refusing: {p.name} was read on other prompts than the real run")
        out.append((run["surrogate_seed"], _layers(run)))
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    rows, per_layer = [], []
    print("section 6.2 -- per model, means over the fitted layers")
    print(f"{'model':>14} {'d':>5} {'prompts':>7} {'layers':>6} {'K(stream)':>12} {'mean':>5} "
          f"{'K/d':>6} {'outside%':>14} {'energy share':>13} {'mean dK':>8} {'mean|dK|':>8} "
          f"{'+/-':>6} {'surr dK':>8} {'mean|real-surr|':>15}")
    for name, d, real_file, tag in MODELS:
        run = _load(RESULTS / real_file)
        real = _layers(run)
        seeds = surrogates(tag, run["prompt_fingerprint"])
        L = sorted(real)
        ks = np.array([real[l]["K_stream"] for l in L])
        dk = np.array([removed(real[l]) for l in L])
        out = np.array([real[l]["outside_fraction"] for l in L]) * 100
        es = np.array([real[l]["j_space_energy_share"] for l in L]) * 100
        s0 = dict(seeds)[0]
        sd = np.array([removed(s0[l]) for l in L])
        print(f"{name:>14} {d:>5} {run['prompts']:>7} {len(L):>6} {ks.min():>5.1f}-{ks.max():<6.1f} "
              f"{ks.mean():>5.1f} {ks.mean() / d:>6.3f} {out.min():>6.1f}-{out.max():<6.1f}% "
              f"{es.min():>5.1f}-{es.max():<5.1f}% {dk.mean():>+8.2f} {np.abs(dk).mean():>8.2f} "
              f"{int((dk > 0).sum()):>3}/{int((dk < 0).sum()):<2} {sd.mean():>+8.2f} "
              f"{np.abs(dk - sd).mean():>15.2f}")
        rows.append((d, ks.mean()))
        if len(seeds) > 1:
            per_layer.append((name, real, seeds))

    w = np.array(rows)
    rho = np.corrcoef(_ranks(w[:, 0]), _ranks(w[:, 1]))[0, 1]
    slope = np.polyfit(np.log(w[:, 0]), np.log(w[:, 1]), 1)[0]
    print(f"\nwidth against mean K(stream), six models: rank correlation {rho:+.3f}, "
          f"log-log slope {slope:+.3f}")

    for name, real, seeds in per_layer:
        shared = sorted(set(real).intersection(*[set(s) for _, s in seeds]))
        tags = " / ".join(str(k) for k, _ in seeds)
        print(f"\nsection 6.1 -- {name}, the layers every surrogate seed covers "
              f"(surrogate seeds {tags})")
        print(f"{'layer':>5} {'K(stream)':>9} {'K(compl)':>8} {'removed':>13} "
              f"{'surrogate removed':>22} {'range':>6} {'excess':>7} {'carried':>7} {'outside%':>8} "
              f"{'energy':>7}")
        for l in shared:
            x = real[l]
            pp = np.array([removed(p) for p in x["per_prompt"]])
            sv = [removed(s[l]) for _, s in seeds]
            print(f"{l:>5} {x['K_stream']:>9.1f} {x['K_complement']:>8.1f} "
                  f"{pp.mean():>6.1f} +- {pp.std(ddof=1):<4.1f} "
                  f"{' / '.join(f'{v:.1f}' for v in sv):>22} {max(sv) - min(sv):>6.1f} {pp.mean() - max(sv):>+7.1f} "
                  f"{x['K_transport_dirs']:>7.1f} {100 * x['outside_fraction']:>7.1f}% "
                  f"{100 * x['j_space_energy_share']:>6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
