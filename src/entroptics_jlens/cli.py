"""``entroptics-jlens`` -- reading a linear readout with its identity separated out.

A Jacobian lens carries a large identity component at depth, and a spectral floor derived from
the matrix it judges is raised by it. Removing it first changes the resolved count by **2x to
21x**, in 9 of 10 published models across five families. ``audit`` prints both columns side by
side so
the gap is a thing you see rather than a thing you have to know about.

    entroptics-jlens catalog              what lenses are published, and how big
    entroptics-jlens fetch gpt2           download one (transports only; no model weights)
    entroptics-jlens audit LENS.pt        the identity share and both rank columns, per layer
    entroptics-jlens compare A.pt B.pt    are two fits the same map?
    entroptics-jlens coverage LENS.pt     how much of a real stream the transport reaches

``audit`` and ``compare`` read the checkpoint and nothing else -- no model, no corpus, no GPU,
no labels. ``coverage`` additionally needs residual streams, which come from a forward pass the
caller runs; see ``--streams``.

Every command refuses with a message naming what it found rather than substituting a default.
A lens file that is really a fitting checkpoint, a layer that was never fitted, a stream array
whose depth does not match the lens: each is a statement about the input, and inventing a value
for it would put a fabricated number under a real heading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import catalog as _catalog
from .coverage import coverage as _coverage
from .decompose import WORTH_DECOMPOSING, decompose, screen
from .io import LensFormatError, load_lens
from .spectra import (energy_spectrum, gram_spectrum, participation_ratio,
                      principal_angles, residual_gram)
from .transport import floor_is_resolvable, transport_spectrum

#: A canonical cosine at or above this counts as a direction the two readouts share. 0.9 is
#: cos(26 deg): well inside "the same direction" and well outside what unaligned subspaces give
#: (two random k-subspaces of R^2560 have essentially all their cosines near 0).
AGREE_AT = 0.9

#: Verdict boundaries on the MEAN canonical cosine, placed in the gap between two measured
#: populations rather than chosen: two published fits of one model read 0.859-0.9985 across
#: depth, and two unrelated maps of the same construction read 0.166-0.191.
SAME_MAP = 0.8
DRIFTED = 0.4

#: Shown under `--help`. Held as a triple-quoted constant rather than built by concatenation so
#: the layout in this file is the layout a reader sees.
EPILOG = """start here:

  entroptics-jlens fetch gpt2      13 MB, transports only, no model weights
  entroptics-jlens audit lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt

audit and compare read the checkpoint and nothing else: no model, no corpus, no GPU,
no labels. coverage additionally needs residual streams, which you collect from a
forward pass; see `coverage --help`.
"""


def _layers(spec: str | None, fitted: list[int]) -> list[int]:
    """Resolve a ``--layers`` argument against the layers a lens actually carries."""
    if spec is None or spec == "all":
        return list(fitted)
    want = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            want.extend(range(int(lo), int(hi) + 1))
        else:
            want.append(int(part))
    missing = [l for l in want if l not in fitted]
    if missing:
        raise ValueError(f"--layers names {missing}, which this lens does not carry; "
                         f"fitted layers are {fitted}")
    return want


def _runs(layers: list[int]) -> str:
    """Layers as contiguous runs -- "0-5, 15-19" rather than a bare min..max, which would read
    as a solid range when the set is a scatter."""
    if not layers:
        return "none"
    out, start, prev = [], layers[0], layers[0]
    for l in layers[1:]:
        if l != prev + 1:
            out.append((start, prev))
            start = l
        prev = l
    out.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def _paired_spectra(J: np.ndarray, dec) -> tuple[np.ndarray, np.ndarray]:
    """Spectra of ``J`` and ``J - alpha*I`` from a single Gram matrix."""
    gram = J.T @ J
    return gram_spectrum(gram), gram_spectrum(residual_gram(gram, J, dec.alpha))


def _read_streams(path: Path) -> list[np.ndarray]:
    """Residual streams as a list of ``(n_layers + 1, T, d)`` arrays, one per prompt."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            out = [np.asarray(z[k]) for k in z.files]
    else:
        arr = np.asarray(np.load(path))
        out = [arr] if arr.ndim == 3 else list(arr)
    bad = [a.shape for a in out if a.ndim != 3]
    if bad or not out:
        raise ValueError(
            f"{path}: expected one or more (n_layers + 1, T, d) arrays of residual streams; "
            f"got {len(out)} array(s)" + (f" with shapes {bad}" if bad else ""))
    return out


# ---------------------------------------------------------------- audit

def cmd_audit(a) -> int:
    lens = load_lens(a.lens)
    layers = _layers(a.layers, lens.source_layers)
    print(f"{a.lens}")
    print(f"  d_model {lens.d_model}   fitted layers {len(lens.source_layers)}   "
          f"n_prompts {lens.n_prompts or 'unrecorded'}")
    print(f"  reading {len(layers)} layer(s) at far={a.far}\n")
    print(f"{'layer':>6}{'alpha':>9}{'identity':>10}{'PR(J)':>10}{'PR(J-aI)':>10}"
          f"{'K(J)':>12}{'K(J-aI)':>12}  reads as")
    print(f"{'':>45}{'mp/fence':>12}{'mp/fence':>12}")

    rows = []
    exact_rows = 0
    for l in layers:
        J = lens.jacobian(l)
        dec = decompose(J, kind="identity")
        # One Gram for both matrices. Two full SVDs is the obvious spelling and pays for the
        # same work twice: `(J - aI)^T(J - aI) = J^T J - a(J + J^T) + a^2 I`, so the second Gram
        # is an O(d^2) update rather than a second matmul, and eigvalsh on a Gram beats an SVD on
        # the matrix. `_floor_is_resolvable` then checks that squaring has not cost the floor its
        # meaning; if it has, the exact route is taken for that layer and only that layer.
        s_j, s_m = _paired_spectra(J, dec)
        if not (floor_is_resolvable(J, s_j, far=a.far)
                and floor_is_resolvable(dec.residual, s_m, far=a.far)):
            exact_rows += 1
            s_j = np.linalg.svd(J, compute_uv=False)
            s_m = np.linalg.svd(dec.residual, compute_uv=False)
        j_mp = transport_spectrum(J, far=a.far, null="mp", s=s_j)
        j_rob = transport_spectrum(J, far=a.far, null="robust", s=s_j)
        m_mp = transport_spectrum(dec.residual, far=a.far, null="mp", s=s_m)
        m_rob = transport_spectrum(dec.residual, far=a.far, null="robust", s=s_m)
        pr_j = participation_ratio(s_j)
        pr_m = participation_ratio(s_m)
        # The verdict reads the identity-free transport, because that is the map. On raw J the
        # identity flattens the spectrum, lifts the self-estimated floor with it, and buries the
        # weak modes underneath: at Qwen3.5-4B layer 30, K reads 25 on J and 183 on J - alpha*I.
        if m_mp.K == 0 and m_rob.K == 0:
            verdict = "resolves nothing"
        elif dec.identity_dominated:
            verdict = "identity-dominated"
        elif m_mp.saturated:
            verdict = "saturated"
        else:
            verdict = "transport"
        rows.append({"layer": l, "alpha": dec.alpha, "identity_energy": dec.removed_energy,
                     "pr_J": pr_j, "pr_residual": pr_m,
                     "K_J_mp": j_mp.K, "K_J_robust": j_rob.K,
                     "K_mp": m_mp.K, "K_robust": m_rob.K, "verdict": verdict})
        print(f"{l:>6}{dec.alpha:>9.3f}{dec.removed_energy:>10.3f}{pr_j:>10.1f}{pr_m:>10.1f}"
              f"{f'{j_mp.K}/{j_rob.K}':>12}{f'{m_mp.K}/{m_rob.K}':>12}  {verdict}", flush=True)

    dom = [r["layer"] for r in rows if r["verdict"] == "identity-dominated"]
    empty = [r["layer"] for r in rows if r["verdict"] == "resolves nothing"]
    print()
    if dom:
        print(f"{len(dom)} of {len(rows)} layers are identity-dominated (layers "
              f"{_runs(dom)}): more than half of ||J||_F^2 there is the residual stream's")
        print("skip connection. A spectral read of J at those layers describes the architecture,")
        print("not the map -- read the J-aI columns, not the J ones.")
        # Said with the layer's own numbers rather than in general. The identity flattens the
        # spectrum and lifts the floor that is estimated from it, so the raw-J rank is not merely
        # different, it is biased downward exactly where the identity is largest.
        worst = max((r for r in rows if r["verdict"] == "identity-dominated"),
                    key=lambda r: r["K_mp"] - r["K_J_mp"])
        if worst["K_mp"] > worst["K_J_mp"]:
            print(f"Concretely, at layer {worst['layer']}: {worst['K_mp']} directions resolved "
                  f"with the identity removed,")
            print(f"and {worst['K_J_mp']} with it still there. Same matrix, same null, same far.")
    else:
        print("no layer is identity-dominated; PR(J) and PR(J-aI) should track each other.")
    if empty:
        print(f"\n{len(empty)} layer(s) resolve no mode above the floor under either null: "
              f"{_runs(empty)}.")
        print("Their PR is still large, and that is not a contradiction. Participation ratio is")
        print("threshold-free -- it reports how flat a spectrum is, and a matrix of pure noise is")
        print("the flattest thing there is. Read PR only where K says something is there.")

    # Restricted to layers that resolve, for exactly that reason: an unrestricted argmax over PR
    # picks the emptiest layer in the file. Measured on the synthetic lens, where the planted
    # band reads PR 17.4 and the noise-only layers outside it read 80.
    # Gated on the mp count, and the K quoted is mp's. The fence is not a null and cannot answer
    # "is anything there"; using it here would let a spectrum's shape decide which layers count
    # as carrying.
    carrying = [r for r in rows if r["K_mp"] > 0]
    if carrying:
        peak = max(carrying, key=lambda r: r["pr_residual"])
        print(f"\namong the {len(carrying)} layer(s) that resolve anything, effective rank peaks "
              f"at layer {peak['layer']}")
        print(f"(PR(J-aI) = {peak['pr_residual']:.1f}, K = {peak['K_mp']} under mp), relative "
              f"depth {peak['layer'] / max(1, max(lens.source_layers)):.2f}.")
    else:
        print("\nno layer resolves a mode, so there is no band to report. On a corpus-averaged")
        print("transport this is expected rather than a failure: averaging removes the noise bulk")
        print("a detection floor needs an edge of. Read PR(J-aI) and compare fits instead.")

    # The two K figures are not two opinions about one quantity, and printing them adjacently
    # invites reading them as one. Only the first answers "how many modes would noise produce".
    print(f"\nEach K is `mp/fence`. Only mp is a calibrated null, and only mp responds to --far "
          f"(={a.far});")
    print("`fence` is the Tukey outlier fence Q3 + 1.5*IQR of the spectrum, which takes no")
    print("false-alarm rate and reads FEWER modes the more heavy-tailed the spectrum is. Read the")
    print("gap between them as a statement about the spectrum's shape, not as an interval on K.")

    if a.json:
        _write_json(a.json, {"command": "audit", "lens": str(a.lens), "d_model": lens.d_model,
                             "n_prompts": lens.n_prompts, "far": a.far,
                             "fitted_layers": lens.source_layers, "layers": rows})
    return 0


# ---------------------------------------------------------------- screen

def cmd_screen(a) -> int:
    """The sweep: every layer's identity share, no SVD anywhere.

    The identity share is what predicts whether removing it changes the answer, and it costs
    O(d^2) against a spectrum's O(d^3). Measured on Qwen3.5-4B: 4.3 s to screen all 31 layers
    against ~93 s to audit them, and it flags the 10 that matter.
    """
    lens = load_lens(a.lens)
    layers = _layers(a.layers, lens.source_layers)
    rows = screen(lens, layers, threshold=a.threshold)

    print(f"{a.lens}")
    print(f"  d_model {lens.d_model}   screening {len(layers)} layer(s) at "
          f"threshold {a.threshold}")
    print()
    print(f"{'layer':>6}{'alpha':>9}{'identity':>10}  needs the full read?")
    for r in rows:
        mark = "yes" if r["worth_decomposing"] else "no"
        print(f"{r['layer']:>6}{r['alpha']:>9.3f}{r['identity']:>10.3f}  {mark}")

    flagged = [r["layer"] for r in rows if r["worth_decomposing"]]
    print()
    if flagged:
        print(f"{len(flagged)} of {len(rows)} layers carry enough identity to change the answer: "
              f"{_runs(flagged)}")
        print()
        print(f"  entroptics-jlens audit {a.lens} --layers "
              f"{','.join(str(x) for x in flagged)}")
    else:
        print(f"No layer reaches an identity share of {a.threshold}. Below that the corrected and")
        print("both reads agree, so the raw transport is the same answer and the SVDs would buy")
        print("nothing. Audit anyway if you want the spectra for another reason.")
    print()
    print("This read does no SVD. The share is exact -- tr(J)/d and one Frobenius norm -- and it")
    print("is the quantity that predicts the change rather than a proxy for it.")

    if a.json:
        _write_json(a.json, {"command": "screen", "lens": str(a.lens),
                             "threshold": a.threshold, "layers": rows})
    return 0


# ---------------------------------------------------------------- compare

def cmd_compare(a) -> int:
    left, right = load_lens(a.left), load_lens(a.right)
    if left.d_model != right.d_model:
        raise ValueError(
            f"{a.left} is d_model {left.d_model} and {a.right} is {right.d_model}. Two transports "
            f"of different width are maps on different spaces; there is no subspace angle between "
            f"them. Comparing across widths needs a shared basis -- see the coverage read.")
    shared = [l for l in left.source_layers if l in right.source_layers]
    if not shared:
        raise ValueError(f"no layer is fitted in both lenses: {left.source_layers} vs "
                         f"{right.source_layers}")
    layers = _layers(a.layers, shared)
    k = int(a.k)
    print(f"left  {a.left}\nright {a.right}")
    print(f"  d_model {left.d_model}   layers fitted in both: {len(shared)}   "
          f"comparing top-{k} subspaces\n")
    print(f"{'layer':>6}{'agree_to':>10}{'cos_mean':>10}{'cos_min':>9}{'dPR(J-aI)':>12}"
          f"{'d identity':>12}  reads as")

    rows = []
    for l in layers:
        A, B = left.jacobian(l), right.jacobian(l)
        da, db = decompose(A), decompose(B)
        ma, mb = da.residual, db.residual
        c = principal_angles(ma, mb, k)
        pa, pb = participation_ratio(energy_spectrum(ma)), participation_ratio(energy_spectrum(mb))
        d_pr = (pb - pa) / pa if pa > 0 else float("nan")
        # How far into the spectrum the two still agree: the length of the leading prefix whose
        # canonical cosines all clear AGREE_AT. This is the question principal angles are for,
        # and it is a measurement rather than a threshold on a summary.
        below = np.flatnonzero(c < AGREE_AT)
        k_agree = int(below[0]) if below.size else k
        # The verdict reads the MEAN, not the minimum. The minimum over k directions is a
        # worst-case that two honest fits of one model do not survive: the two published
        # Qwen3.5-4B fits (n=417, n=1000) read cos_min 0.0004 at layer 0 over k=400, because the
        # 400th direction sits in the tail where fit noise dominates -- while their cos_mean over
        # the same directions is 0.859 to 0.9985 across depth. Two unrelated maps of the same
        # shape measure 0.166 and 0.191. SAME_MAP sits in the gap between those two measured
        # populations; it is not tuned, and both are recorded in tests/test_cli.py.
        verdict = ("same map" if float(c.mean()) > SAME_MAP else
                   "drifted" if float(c.mean()) > DRIFTED else "different map")
        rows.append({"layer": l, "k_agree": k_agree, "agree_at": AGREE_AT,
                     "cos_mean": float(c.mean()), "cos_min": float(c.min()),
                     "pr_left": pa, "pr_right": pb, "pr_relative_change": d_pr,
                     "identity_left": da.removed_energy, "identity_right": db.removed_energy,
                     "verdict": verdict})
        print(f"{l:>6}{k_agree:>10}{c.mean():>10.4f}{c.min():>9.4f}{d_pr:>+12.1%}"
              f"{db.removed_energy - da.removed_energy:>+12.3f}  {verdict}", flush=True)

    print()
    worst = min(rows, key=lambda r: r["cos_mean"])
    print(f"the two agree least at layer {worst['layer']}: mean principal cosine "
          f"{worst['cos_mean']:.4f} over the top {k} directions, and they hold above "
          f"{AGREE_AT} for {worst['k_agree']} of them.")
    print(f"Read agree_to, not cos_min. The smallest of {k} canonical cosines lands in the tail")
    print("of the spectrum, where two honest fits of one model disagree freely; how far the")
    print("agreement extends is the answer a scalar cannot give.")
    print("\nThis read sees the map only -- it cannot say which of the two is better. For that,")
    print("run `coverage` against real streams.")
    if a.json:
        _write_json(a.json, {"command": "compare", "left": str(a.left), "right": str(a.right),
                             "k": k, "layers": rows})
    return 0


# ---------------------------------------------------------------- coverage

def cmd_coverage(a) -> int:
    lens = load_lens(a.lens)
    layers = _layers(a.layers, lens.source_layers)
    streams = _read_streams(a.streams)
    depth = streams[0].shape[0]
    widths = {int(s.shape[2]) for s in streams}
    if widths != {lens.d_model}:
        raise ValueError(f"{a.streams}: streams are d={sorted(widths)} and the lens is "
                         f"d_model={lens.d_model}; a coverage read needs both on one basis")
    # Layer l's transport carries the stream ENTERING layer l, which hidden_states indexes at
    # l + 1 (index 0 is the embedding output). Reading s[l] instead is an off-by-one that still
    # produces a plausible curve, so it is checked rather than assumed.
    too_deep = [l for l in layers if l + 1 >= depth]
    if too_deep:
        raise ValueError(f"layers {too_deep} need hidden state index l+1, but the streams carry "
                         f"only {depth} (indices 0..{depth - 1}). Collect streams from the same "
                         f"model the lens was fitted on.")
    rng = np.random.default_rng(a.seed)
    tokens = [int(s.shape[1]) for s in streams]
    print(f"{a.lens}\n  {len(streams)} stream(s), {min(tokens)}-{max(tokens)} tokens each, "
          f"d={lens.d_model}, {'raw J' if a.raw else 'J - alpha*I'}, far={a.far}")
    # A (T, d) frame has rank at most T, so a stream shorter than the residual width cannot
    # resolve more directions than it has tokens however much structure the model put there.
    # Said here rather than left to be inferred: a 3-token prompt reads k_signal = 1 and reports
    # a one-dimensional overlap at several hundred times chance, which looks like a strong result
    # and is a statement about the prompt. No threshold is involved -- T < d is the whole test.
    if min(tokens) < lens.d_model:
        print(f"\n  NOTE: {sum(1 for t in tokens if t < lens.d_model)} of {len(streams)} stream(s)"
              f" carry fewer tokens than d={lens.d_model}. A (T, d) frame has rank at most T, so"
              f"\n  the k_sig column below is capped by token count, not by the stream's"
              f" structure. Collect\n  sequences of at least d tokens before reading these"
              f" numbers as being about the model.")
    print()
    print(f"{'layer':>6}{'k_sig':>7}{'k_read':>8}{'coverage':>10}{'chance':>9}{'x chance':>10}"
          f"{'random map':>12}{'x chance':>10}")

    rows = []
    for l in layers:
        J0 = lens.jacobian(l)
        J = J0 if a.raw else decompose(J0, kind="identity").residual
        real, null, ctrl, ctrl_null, ks, kt = [], [], [], [], [], []
        for s in streams:
            H = np.asarray(s[l + 1], dtype=np.float64)
            c = _coverage(H, H @ J.T, far=a.far)
            real.append(c.coverage)
            null.append(c.null)
            ks.append(c.k_signal)
            kt.append(c.k_readout)
            q = np.linalg.qr(rng.standard_normal((H.shape[1], max(1, c.k_readout))))[0]
            cx = _coverage(H, (H @ q) @ q.T, far=a.far)
            ctrl.append(cx.coverage)
            ctrl_null.append(cx.null)
        row = {"layer": l, "k_signal": float(np.mean(ks)), "k_readout": float(np.mean(kt)),
               "coverage": float(np.mean(real)), "chance": float(np.mean(null)),
               "random_map": float(np.mean(ctrl)), "random_chance": float(np.mean(ctrl_null))}
        # A ratio to a zero chance level is not a large number, it is an absent one. It happens
        # when the stream or the readout image resolves nothing above its own floor, and the row
        # is reported as unread rather than as nan -- a printed nan under a real heading is the
        # one thing this package will not do.
        row["read"] = row["chance"] > 0.0 and row["k_signal"] > 0.0
        row["ratio"] = row["coverage"] / row["chance"] if row["read"] else None
        row["random_ratio"] = (row["random_map"] / row["random_chance"]
                               if row["read"] and row["random_chance"] else None)
        rows.append(row)
        if row["read"]:
            print(f"{l:>6}{row['k_signal']:>7.0f}{row['k_readout']:>8.0f}"
                  f"{row['coverage']:>10.4f}{row['chance']:>9.4f}{row['ratio']:>10.1f}"
                  f"{row['random_map']:>12.4f}{row['random_ratio']:>10.1f}", flush=True)
        else:
            print(f"{l:>6}{row['k_signal']:>7.0f}{row['k_readout']:>8.0f}"
                  f"{'-':>10}{'-':>9}{'-':>10}{'-':>12}{'-':>10}  nothing resolved", flush=True)

    read = [r for r in rows if r["read"]]
    print()
    if not read:
        print("no layer could be read: at every one, either the stream or the transported frame")
        print("resolved no mode above its own noise floor, so there is no subspace to overlap.")
        print("A coverage of 0 against a chance of 0 is not a small overlap; it is no measurement.")
    else:
        ratio = [r["ratio"] for r in read]
        ctrl_ratio = [r["random_ratio"] for r in read if r["random_ratio"] is not None]
        print(f"real        {min(ratio):.0f}x - {max(ratio):.0f}x chance")
        if ctrl_ratio:
            print(f"random map  {min(ctrl_ratio):.1f}x - {max(ctrl_ratio):.1f}x chance"
                  f"   (chance is 1.0x)")
        if len(read) < len(rows):
            print(f"\n{len(rows) - len(read)} of {len(rows)} layers resolved nothing and are "
                  f"excluded from that range.")
        print("\nThe random-map column is the control, not decoration: it is a map of the same")
        print("rank on the same frame, and it is what a read has to beat to mean anything.")
    if a.raw:
        print("\n--raw reads J whole. The identity core covers everything trivially, so this")
        print("curve climbs with depth whatever the transport does. Prefer the default.")
    if a.json:
        _write_json(a.json, {"command": "coverage", "lens": str(a.lens),
                             "streams": str(a.streams), "n_streams": len(streams),
                             "raw": bool(a.raw), "far": a.far, "seed": a.seed, "layers": rows})
    return 0


# ---------------------------------------------------------------- catalogue

def cmd_catalog(a) -> int:
    print(f"{'key':<16}{'MB':>9}  {'revision':<12} path in {_catalog.REPO}")
    for key, (_sub, _fn, rev, mb) in _catalog.CATALOG.items():
        print(f"{key:<16}{mb:>9.1f}  {rev:<12} {_catalog.repo_path(key)}")
    return 0


def cmd_fetch(a) -> int:
    # Every key is checked before anything is downloaded. Otherwise a typo in the third of three
    # names is found after two multi-gigabyte downloads have already run.
    unknown = [k for k in a.keys if k not in _catalog.CATALOG]
    if unknown:
        raise ValueError(f"unknown lens key(s) {unknown}; run `entroptics-jlens catalog` for the "
                         f"{len(_catalog.CATALOG)} published lenses")
    for key in a.keys:
        _sub, _fn, rev, mb = _catalog.CATALOG[key]
        print(f"fetching {key} (~{mb:.0f} MB, revision {rev}) ...", flush=True)
        print(f"  -> {_catalog.fetch(key, a.dir)}", flush=True)
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="entroptics-jlens",
        # Written out rather than taken from __doc__: the module docstring is reStructuredText and
        # its markup renders as literal backticks in a terminal.
        description="A Jacobian lens carries a large identity component at depth, and a "
                    "spectral floor derived from the matrix it judges is raised by it. "
                    "Removing it first changes the resolved count by 2x to 21x, across 9 of "
                    "10 published models. `audit` prints both columns side by side.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="what each layer's transport is, from the file alone")
    p.add_argument("lens", type=Path, help="a lens checkpoint saved by jlens, or fetched")
    p.add_argument("--layers", default="all", help="'all' (default), or 0,5,10 or 0-12")
    p.add_argument("--far", type=float, default=0.05, help="false-alarm rate for the noise floor")
    p.add_argument("--json", type=Path, help="also write the table here")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("screen", help="which layers need the expensive read? no SVD")
    p.add_argument("lens", type=Path, help="a lens checkpoint")
    p.add_argument("--layers", default="all")
    p.add_argument("--threshold", type=float, default=WORTH_DECOMPOSING,
                   help=f"identity share above which a layer is flagged (default "
                        f"{WORTH_DECOMPOSING})")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("compare", help="are two lenses of one model the same map?")
    p.add_argument("left", type=Path, help="a lens checkpoint")
    p.add_argument("right", type=Path, help="another, of the same model width")
    p.add_argument("--layers", default="all")
    p.add_argument("--k", type=int, default=64, help="how far into the spectrum to compare")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("coverage", help="how much of a real stream the transport reaches")
    p.add_argument("lens", type=Path, help="a lens checkpoint")
    p.add_argument("--streams", type=Path, required=True,
                   help=".npy/.npz of (n_layers + 1, T, d) residual streams, one per prompt")
    p.add_argument("--layers", default="all")
    p.add_argument("--raw", action="store_true", help="read J whole, identity core included")
    p.add_argument("--far", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0, help="for the random-map control")
    p.add_argument("--json", type=Path)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("catalog", help="the published lenses, with sizes")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("fetch", help="download a published lens (transports only)")
    p.add_argument("keys", nargs="+", help="catalogue keys; run `catalog` to list them")
    p.add_argument("--dir", type=Path, default=Path("lenses"))
    p.set_defaults(func=cmd_fetch)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (FileNotFoundError, KeyError, ValueError, LensFormatError, ImportError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
