"""Experiment 1 -- the resolved rank of every transport in a fitted lens.

The cheapest question worth asking, and the only one that needs no model, no corpus and no
forward pass: read ``K_l``, the rank ``J_l`` resolves against the derived Tracy-Widom edge,
straight off a published ``lens.pt``.

What the answer decides:

  * ``K_l`` on the order of tens through a middle band, collapsing at the ends, is an
    independent parameter-free derivation of the reported workspace capacity and its layer
    band from an artefact already published.
  * ``K_l`` far larger, or monotone in depth, says the reported figure comes from the chosen
    activity threshold rather than from the transport.

Either is a result. Neither needs a GPU.

Usage
-----
    python experiments/fetch_lens.py --list
    python experiments/fetch_lens.py qwen3.5-4b
    python experiments/exp1_transport_spectrum.py --lens lenses/.../lens.pt --layers all \
        --controls 10,20,30 --live results/live.html --out results/exp1.json

``--layers`` is required and takes an explicit list or ``all``: one SVD per layer at
``d_model`` is real CPU time, and a default that quietly read a subset would report a curve
with holes in it as if it were the curve.

``--live`` writes a self-contained page after every layer, meta-refreshing while the run is in
flight. Open it once and watch the curve fill in.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
import live_report                                                     # noqa: E402


def parse_layers(spec: str, available: list[int]) -> list[int]:
    if spec.strip() == "all":
        return list(available)
    want = [int(v) for v in spec.replace(" ", "").split(",") if v]
    missing = [l for l in want if l not in available]
    if missing:
        raise SystemExit(f"refusing: layers {missing} are not fitted in this lens; "
                         f"available layers are {available}")
    return want


def sha256(path: Path, limit: int = 1 << 24) -> str:
    """Digest of the first 16 MiB -- enough to identify the artefact, cheap on a 10 GB file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def read_layer(lens, layer: int, far: float, seed: int, control_seed: int | None) -> dict:
    t0 = time.time()
    J = lens.jacobian(layer)
    under = je.spectrum_under_nulls(J, far=far, seed=seed)
    mp, rb = under["mp"], under["robust"]
    row = {"layer": layer,
           "K_mp": mp.K, "K_robust": rb.K,
           "floor_mp": mp.floor, "floor_robust": rb.floor,
           "energy_resolved_mp": mp.energy_resolved,
           "energy_resolved_robust": rb.energy_resolved,
           "saturated": mp.saturated or rb.saturated,
           "sigma_top": float(mp.singular[0]),
           "sigma_median": float(np.median(mp.singular)),
           # The number that says whether the edge's Gaussian null applies at all. Measured
           # 81 / 709 / 7.4e4 at gpt2 layers 0/5/10 -- so on a real transport K must be read
           # against the shuffled baseline, never against 0.
           "excess_kurtosis": mp.excess_kurtosis,
           "K_mp_far_1e-3": mp.recount(1e-3), "K_mp_far_0.2": mp.recount(0.2)}
    if control_seed is not None:
        rng = np.random.default_rng(control_seed)
        row["controls"] = {}
        for name, M in (("gaussian", je.gaussian_null(J.shape, rng, sigma=je.frobenius_sigma(J))),
                        ("shuffled", je.shuffled_entries(J, rng)),
                        ("matched_spectrum", je.matched_spectrum(J, rng))):
            c = je.spectrum_under_nulls(M, far=far, seed=seed)
            row["controls"][name] = {"K_mp": c["mp"].K, "K_robust": c["robust"].K}
    row["seconds"] = time.time() - t0
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lens", required=True, type=Path, help="a lens.pt saved by jlens")
    ap.add_argument("--layers", required=True,
                    help="'all' or a comma-separated list; explicit by design")
    ap.add_argument("--far", type=float, default=0.05, help="false-alarm level (the reader's)")
    ap.add_argument("--controls", default="",
                    help="comma-separated layers to run the three nulls on (default: none)")
    ap.add_argument("--seed", type=int, default=0, help="control RNG seed; recorded in the run")
    ap.add_argument("--live", type=Path, default=None,
                    help="self-contained page, rewritten after every layer")
    ap.add_argument("--out", type=Path, default=None, help="write the run as JSON")
    ap.add_argument("--name", default=None, help="label for the page (default: the file stem)")
    a = ap.parse_args(argv)

    lens = je.load_lens(a.lens)                     # refuses loudly; no stand-in transport
    layers = parse_layers(a.layers, lens.source_layers)
    controls = set(parse_layers(a.controls, lens.source_layers)) if a.controls.strip() else set()
    name = a.name or a.lens.stem.replace("_jacobian_lens", "")

    print(f"lens      {a.lens}\n          d_model={lens.d_model}  n_prompts={lens.n_prompts}  "
          f"layers={len(lens)}  mmapped={lens.mmapped}", flush=True)
    print(f"reading   {len(layers)} layer(s) at far={a.far}; "
          f"controls on {sorted(controls) if controls else 'NO layers'}", flush=True)
    if not controls:
        print("NOTE      no control was run. K alone is not a measurement -- rerun with "
              "--controls before the numbers are quoted anywhere.", flush=True)
    print("NOTE      the two providers estimate the noise variance differently and neither "
          "dominates\n          on real transports. Quote the pair, never one of them.", flush=True)

    meta = {"name": name, "path": str(a.lens), "d_model": lens.d_model,
            "n_layers": len(layers), "n_prompts": lens.n_prompts, "far": a.far, "elapsed": 0.0}
    if a.live:
        live_report.render(a.live, meta, [], done=False)
        print(f"live      {a.live.resolve().as_uri()}", flush=True)

    print(f"\n{'layer':>6} {'K_mp':>6} {'K_rob':>6} {'E_mp':>7} {'E_rob':>7} "
          f"{'floor_mp':>10} {'floor_rob':>10} {'sigma_top':>10} {'kurt':>10} {'s':>6}  "
          f"controls K_mp/K_rob (gauss, shuf, matched)", flush=True)

    rows, t_start = [], time.time()
    for layer in layers:
        row = read_layer(lens, layer, a.far, a.seed, a.seed if layer in controls else None)
        rows.append(row)
        c = row.get("controls")
        ctl = ("  ".join(f"{c[k]['K_mp']}/{c[k]['K_robust']}"
                         for k in ("gaussian", "shuffled", "matched_spectrum")) if c else "")
        flag = "  SATURATED" if row["saturated"] else ""
        print(f"{layer:>6} {row['K_mp']:>6} {row['K_robust']:>6} "
              f"{row['energy_resolved_mp']:>7.4f} {row['energy_resolved_robust']:>7.4f} "
              f"{row['floor_mp']:>10.4g} {row['floor_robust']:>10.4g} "
              f"{row['sigma_top']:>10.4g} {row['excess_kurtosis']:>10.4g} "
              f"{row['seconds']:>6.1f}  {ctl}{flag}", flush=True)
        if a.live:
            meta["elapsed"] = time.time() - t_start
            live_report.render(a.live, meta, rows, done=False)

    meta["elapsed"] = time.time() - t_start
    run = {"lens": {"path": str(a.lens), "sha256_head": sha256(a.lens),
                    "d_model": lens.d_model, "n_prompts": lens.n_prompts,
                    "source_layers": lens.source_layers},
           "far": a.far, "seed": a.seed, "control_layers": sorted(controls),
           "elapsed_seconds": meta["elapsed"],
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "platform": platform.platform()},
           "layers": rows}
    if a.live:
        live_report.render(a.live, meta, rows, done=True)
    if a.out:
        live_report.dump_json(a.out, run)
        print(f"\nwrote {a.out}", flush=True)
    print(f"done      {len(rows)} layers in {meta['elapsed']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
