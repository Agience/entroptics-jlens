"""How much of each headline number is a mean, and how wide is the sample behind it?

sec 7.1 reported that gemma-3-1b's complement removes "exactly" the number of directions its
transport carries -- 12.0 against 12.0. A second prompt sample gave 9.9, and the per-prompt
removals turned out to be 11, 10, 16, 11, 8, 5, 18, 0: a standard deviation of 5.7 around a mean
that had landed on the deterministic value by chance. The identity was an artefact of averaging.

Every quantity exp4 reports is a mean over prompts and every results file keeps the per-prompt
values, so the same question can be asked of all of them at once: for each layer, how large is the
spread relative to the number being quoted? A quantity whose per-prompt standard deviation rivals
its mean is a quantity no single-figure claim should be built on.

Reports the worst offenders per file, and separately the spread on `outside%`, which is the
central claim of sec 7.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

METRICS = ("K_stream", "K_complement", "K_transport_dirs", "certify_residual", "etendue_match")


def spread(per, key):
    """Mean and standard deviation of one metric across a layer's prompts."""
    v = [p[key] for p in per if key in p and p[key] is not None]
    if len(v) < 2:
        return None
    return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path,
                    default=sorted(Path("results").glob("pr_*.json")))
    a = ap.parse_args(argv)

    print(f"{'file':22s}{'metric':20s}{'worst layer':>12}{'mean':>9}{'sd':>8}{'sd/mean':>9}")
    for f in a.files:
        try:
            doc = je.load_complete(f)
        except Exception as exc:
            print(f"{f.name:22s}refused: {type(exc).__name__}")
            continue
        for key in METRICS:
            worst = None
            for row in doc["layers"]:
                per = row.get("per_prompt") or []
                st = spread(per, key)
                if st is None:
                    continue
                m, sd, n = st
                rel = sd / abs(m) if m else float("inf")
                if worst is None or rel > worst[3]:
                    worst = (row["layer"], m, sd, rel)
            if worst:
                l, m, sd, rel = worst
                flag = "  <-- mean is not a summary" if rel > 0.5 else ""
                print(f"{f.name:22s}{key:20s}{l:>12}{m:>9.2f}{sd:>8.2f}{rel:>9.2f}{flag}")

    print()
    print("outside% -- the central claim of sec 7 -- per-prompt, by model")
    print(f"{'file':22s}{'layers':>8}{'mean of means':>15}{'worst sd':>10}{'worst layer':>12}")
    for f in a.files:
        try:
            doc = je.load_complete(f)
        except Exception:
            continue
        means, worst = [], (0.0, None)
        for row in doc["layers"]:
            per = row.get("per_prompt") or []
            v = [p["K_complement"] / p["K_stream"] for p in per
                 if p.get("K_stream")]
            if len(v) < 2:
                continue
            means.append(float(np.mean(v)))
            sd = float(np.std(v, ddof=1))
            if sd > worst[0]:
                worst = (sd, row["layer"])
        if means:
            print(f"{f.name:22s}{len(means):>8}{np.mean(means):>15.1%}"
                  f"{worst[0]:>10.1%}{str(worst[1]):>12}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
