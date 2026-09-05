"""Do the modes the identity hides name the same tokens in two independent fits?

`exp52` shows what the hidden modes name. Showing is not evidence: a reader can always suspect the
coherent-looking ones were picked. This tests them.

The catalogue publishes two fits of Qwen3.5-4B, over 1000 and 417 sequences. They are independent
estimates of the same transport, so a direction that is structure has to appear in both, and a
direction that is estimator noise has no reason to. Decoding each fit's resolved modes through the
model's readout turns that into a statement about tokens: **does the resolved set name the same
vocabulary in both fits, and does the agreement extend exactly as far as the identity-free floor
says it should?**

Compared as a set over a block of modes rather than mode by mode. Singular vectors with close
singular values rotate freely within their subspace between two fits, so a per-mode comparison
measures that rotation; the union of the top tokens over modes 0..K does not move under it.

    python experiments/exp53_hidden_modes_reproduce.py

Needs both Qwen3.5-4B lens files and the Qwen3.5-4B checkpoint in the HuggingFace cache.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from entroptics_jlens.decompose import decompose            # noqa: E402
from entroptics_jlens.io import load_lens                   # noqa: E402
from entroptics_jlens.transport import transport_spectrum   # noqa: E402

DIR = "lenses/qwen3.5-4b/jlens/Salesforce-wikitext"
FIT_A = f"{DIR}/Qwen3.5-4B_jacobian_lens_n1000.pt"
FIT_B = f"{DIR}/Qwen3.5-4B_jacobian_lens.pt"
REPO = "models--Qwen--Qwen3.5-4B"


def readout(repo: str, head_key: str, norm_key: str):
    """The unembedding and final-norm gain, read straight from the cached checkpoint.

    Loaded from safetensors rather than through `AutoModelForCausalLM` so only these two tensors
    are materialised; the full model is an order of magnitude larger and none of the rest is used.
    Kept float32: the decode is one matrix-vector product per mode and float64 would triple the
    footprint for digits that do not reach the token ranking.
    """
    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer
    except ImportError as exc:                                    # pragma: no cover - env
        raise ImportError("needs torch, safetensors and transformers: "
                          "pip install 'entroptics-jlens[lens]' transformers") from exc
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{repo}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {repo}")
    root = Path(snaps[0])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for key in (head_key, norm_key):
        if key not in index:
            raise KeyError(f"{key!r} is not in the checkpoint index; keys look like "
                           f"{sorted(k for k in index if 'norm' in k or 'head' in k)[:6]}")
        with safe_open(str(root / index[key]), framework="pt") as f:
            out[key] = f.get_tensor(key).to(torch.float32).numpy()
    tok = AutoTokenizer.from_pretrained(str(root))
    return tok, out[head_key], out[norm_key]


def top_tokens(U: np.ndarray, upto: int, W_U: np.ndarray, g: np.ndarray, per: int,
               lo: int = 0) -> set[int]:
    """The union of the strongest token ids over modes ``lo..upto``.

    Both ends of each mode are taken: a singular vector's sign is arbitrary, so "promotes" and
    "suppresses" are not properties of the direction.

    ``lo`` selects a block rather than a prefix, which is what the equal-width sweep needs: a
    comparison between two prefixes differs in size as well as position, and one between two
    equal blocks differs only in position.
    """
    ids: set[int] = set()
    for k in range(lo, upto):
        logits = W_U @ (U[:, k].astype(np.float32) * g)
        order = np.argsort(logits)
        ids.update(int(i) for i in order[-per:])
        ids.update(int(i) for i in order[:per])
    return ids


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", default=FIT_A)
    ap.add_argument("--b", default=FIT_B)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--head-key", default="model.language_model.embed_tokens.weight",
                    help="Qwen3.5-4B ties its embeddings, so the readout is the embedding "
                         "matrix; a checkpoint with an untied head needs lm_head.weight")
    ap.add_argument("--norm-key", default="model.language_model.norm.weight")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--per", type=int, default=20, help="tokens taken from each end of each mode")
    ap.add_argument("--blocks", type=int, default=7,
                    help="consecutive equal-width blocks to compare across the spectrum")
    ap.add_argument("--show", type=int, default=0,
                    help="print the tokens for this many modes of the hidden block")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    A, B = load_lens(a.a), load_lens(a.b)
    layer = a.layer if a.layer is not None else max(A.source_layers)
    JA, JB = A.jacobian(layer), B.jacobian(layer)
    MA, MB = decompose(JA).residual, decompose(JB).residual

    K_JA = transport_spectrum(JA).K
    K_MA = transport_spectrum(MA).K
    print(f"Qwen3.5-4B layer {layer}   n={A.n_prompts} vs {B.n_prompts}   d={JA.shape[0]}")
    print(f"K(J) = {K_JA}   K(J - alpha I) = {K_MA}   hidden by the identity: {K_MA - K_JA}\n")

    UA = np.linalg.svd(MA, full_matrices=False)[0]
    UB = np.linalg.svd(MB, full_matrices=False)[0]
    tok, W_U, g = readout(a.repo, a.head_key, a.norm_key)
    print(f"readout {W_U.shape}, gain {g.shape}\n")

    # Blocks chosen around the two floors: what a raw read resolves, what the identity-free read
    # resolves, and the same distance again past it as a control.
    marks = [K_JA, (K_JA + K_MA) // 2, K_MA, 2 * K_MA, 4 * K_MA]
    marks = [m for m in marks if 0 < m <= UA.shape[1]]

    rows = []
    print(f"{'modes 0..K':>12} {'Jaccard(A,B)':>14}   what the block is")
    for m in marks:
        sa, sb = top_tokens(UA, m, W_U, g, a.per), top_tokens(UB, m, W_U, g, a.per)
        j = jaccard(sa, sb)
        label = ("resolved without the correction" if m == K_JA else
                 "inside the identity-free floor" if m <= K_MA else
                 "past the floor")
        rows.append({"upto": m, "jaccard": j, "n_a": len(sa), "n_b": len(sb), "label": label})
        print(f"{m:>12} {j:>14.3f}   {label}")

    # The control that matters: the tokens named by the block PAST the floor, on their own.
    if 2 * K_MA <= UA.shape[1]:
        sa = top_tokens(UA, 2 * K_MA, W_U, g, a.per) - top_tokens(UA, K_MA, W_U, g, a.per)
        sb = top_tokens(UB, 2 * K_MA, W_U, g, a.per) - top_tokens(UB, K_MA, W_U, g, a.per)
        hidden_a = top_tokens(UA, K_MA, W_U, g, a.per) - top_tokens(UA, K_JA, W_U, g, a.per)
        hidden_b = top_tokens(UB, K_MA, W_U, g, a.per) - top_tokens(UB, K_JA, W_U, g, a.per)

        # Chance, drawn rather than derived: two sets of the same sizes taken uniformly from the
        # vocabulary. Without it a Jaccard is a number with no scale, and both blocks below sit
        # well above it -- the comparison that carries the result is between them, not against zero.
        rng = np.random.default_rng(0)
        V = W_U.shape[0]
        def chance(na: int, nb: int, draws: int = 20) -> float:
            return float(np.mean([jaccard(set(rng.choice(V, na, replace=False).tolist()),
                                          set(rng.choice(V, nb, replace=False).tolist()))
                                  for _ in range(draws)]))

        jh, jp = jaccard(hidden_a, hidden_b), jaccard(sa, sb)
        ch, cp = chance(len(hidden_a), len(hidden_b)), chance(len(sa), len(sb))
        print(f"\n{'block':>14} {'Jaccard':>9} {'chance':>8} {'x chance':>9}  sizes")
        print(f"{f'{K_JA}..{K_MA}':>14} {jh:>9.3f} {ch:>8.3f} {jh / ch:>8.1f}x  "
              f"{len(hidden_a)}/{len(hidden_b)}   modes the identity hides")
        print(f"{f'{K_MA}..{2 * K_MA}':>14} {jp:>9.3f} {cp:>8.3f} {jp / cp:>8.1f}x  "
              f"{len(sa)}/{len(sb)}   past the floor")
        print(f"\nthe hidden block reproduces {jh / jp:.2f}x as well as the block past the floor")
        rows.append({"block": f"{K_JA}..{K_MA}", "jaccard": jh, "chance": ch,
                     "n_a": len(hidden_a), "n_b": len(hidden_b),
                     "label": "hidden by the identity, alone"})
        rows.append({"block": f"{K_MA}..{2 * K_MA}", "jaccard": jp, "chance": cp,
                     "n_a": len(sa), "n_b": len(sb), "label": "past the floor, alone"})

    # Consecutive blocks of one width, so a comparison between two of them is not a comparison of
    # where they sit in the spectrum. Two adjacent blocks always differ, because agreement decays
    # with mode index; what the floor would have to produce is a break in that decay.
    width = K_MA - K_JA
    blocks = []
    print(f"\nconsecutive blocks of {width} modes:")
    for i in range(a.blocks):
        lo, hi = K_JA + i * width, K_JA + (i + 1) * width
        if hi > UA.shape[1]:
            break
        j = jaccard(top_tokens(UA, hi, W_U, g, a.per, lo=lo),
                    top_tokens(UB, hi, W_U, g, a.per, lo=lo))
        blocks.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "jaccard": j})
        print(f"  {lo:>5}-{hi:<6} mid {(lo + hi) / 2:>6.0f}   Jaccard {j:.3f}", flush=True)

    fit = None
    if len(blocks) > 2:
        mids = np.array([b["mid"] for b in blocks[1:]])
        js = np.array([b["jaccard"] for b in blocks[1:]])
        slope, const = np.polyfit(np.log(mids), np.log(js), 1)
        pred = float(np.exp(const) * blocks[0]["mid"] ** slope)
        resid = np.log(js) - (slope * np.log(mids) + const)
        r2 = float(1 - np.sum(resid ** 2) / np.sum((np.log(js) - np.log(js).mean()) ** 2))
        obs = blocks[0]["jaccard"]
        fit = {"exponent": float(slope), "r2": r2, "predicted": pred, "observed": obs,
               "excess_pct": 100 * (obs / pred - 1)}
        print(f"\npower law fitted to the blocks past the floor: "
              f"exponent {slope:.3f}, r2 {r2:.4f}")
        print(f"  recovered block predicted {pred:.3f}, observed {obs:.3f}, "
              f"excess {100 * (obs / pred - 1):+.1f}%")

    # The null that matters for a decoded comparison. Any direction pushed through W_U lands on a
    # privileged slice of the vocabulary, so two unrelated directions overlap far more than two
    # uniform token draws do, and a uniform draw understates chance by roughly five times.
    rng2 = np.random.default_rng(0)
    decoded = {}
    print("\ndecode-matched null, random directions through the same readout:")
    for n in sorted({100, width}):
        Ra = np.linalg.qr(rng2.standard_normal((UA.shape[0], n)))[0]
        Rb = np.linalg.qr(rng2.standard_normal((UA.shape[0], n)))[0]
        decoded[str(n)] = jaccard(top_tokens(Ra, n, W_U, g, a.per),
                                  top_tokens(Rb, n, W_U, g, a.per))
        print(f"  {n:>4} directions: Jaccard {decoded[str(n)]:.3f}", flush=True)
    rows.append({"label": "block sweep", "width": width, "blocks": blocks,
                 "power_law": fit, "decoded_null": decoded})

    if a.show:
        # Illustration, from the same model and the same block the number above is about. Fit A
        # only: the claim that both fits name the same vocabulary is the Jaccard, not these rows.
        print(f"\nwhat the hidden block names (fit A, modes {K_JA}..{K_MA}):")
        for k in np.linspace(K_JA, K_MA - 1, a.show, dtype=int):
            logits = W_U @ (UA[:, k].astype(np.float32) * g)
            order = np.argsort(logits)
            hi = [tok.decode([int(i)]) for i in order[-8:][::-1]]
            print(f"  mode {k:>4}  {' | '.join(repr(t) for t in hi)}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"layer": layer, "K_J": K_JA, "K_M": K_MA, "per": a.per, "rows": rows},
            indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
