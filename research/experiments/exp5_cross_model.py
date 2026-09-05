"""Experiment 5 -- two models meeting on one screen.

Everything before this treated the Jacobian lens as an object to audit. That is not what a lens
is for. A ``Lens`` is a conversion that puts a system onto a ``Screen`` where other systems land,
and a ``Screen`` earns its place only when the question is *between* signals. So: two models,
each transported through its own Jacobian into a basis they share, meeting.

**The shared basis has to be the vocabulary.** Qwen3.5-0.8B and Qwen3.5-4B have different
residual widths (1024 and 2560), so no residual basis is common to them. Their token ids are
identical -- all 248,077 of them -- so the readout space is. Each side's conversion is its own:

    entry_A : H_A -> unembed_A(H_A @ J_A^T)[keep]      (1024 -> |keep|)
    entry_B : H_B -> unembed_B(H_B @ J_B^T)[keep]      (2560 -> |keep|)

Both land on ``|keep|`` columns, which is what makes it one screen. Entry-only, by construction:
there is no honest inverse from logits back to a residual, and a fabricated one would put an
invention where a measurement belongs.

**What is measured.** ``Screen.couple`` -- the signed coupling with an exact permutation null.
Not a logit cosine: a score with a null behind it, so "these two models agree" becomes a
statement with a false-alarm level attached. ``coupling`` returns the evidence: the signed z, the
strength, the phase, how tightly the two sides are locked.

**What is deliberately NOT measured here.** ``uncondensed`` -- "what A carries that B cannot
receive" -- is the natural next question and the read for it does not work. It fails its own
null (a random matrix of the same rank returns the same answer, PAPER.md sec 7.0) and it fails
an analytic-coverage benchmark where the overlap is exact by construction
(``tests/test_coverage.py``). Until that is fixed, a cross-model complement would be a number
with nothing behind it.

Usage
-----
    python experiments/exp5_cross_model.py --a Qwen/Qwen3.5-0.8B --b Qwen/Qwen3.5-4B \\
        --lens-a lenses/qwen3.5-0.8b/.../Qwen3.5-0.8B_jacobian_lens.pt \\
        --lens-b lenses/qwen3.5-4b/.../Qwen3.5-4B_jacobian_lens_n1000.pt \\
        --streams-b streams/qwen35_4b_streams.npz --keep 4096 --out results/cross.json
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
from entroptics import Screen                                          # noqa: E402
from exp4_stream_complement import (collect_streams, unembed_fn,       # noqa: E402
                                    wikitext_prompts)

A, B = "model_a", "model_b"


def load_readout(repo: str):
    """Just the final norm and the output head, pulled straight from the checkpoint.

    A crossing needs each side's readout, not its whole forward pass -- and for the larger model
    the streams already exist on disk. Loading 16 GB of weights to use two tensors would be the
    kind of waste that turns a cheap measurement into an expensive one.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json as _json

    idx = None
    try:
        p = hf_hub_download(repo, "model.safetensors.index.json")
        idx = _json.load(open(p))["weight_map"]
    except Exception:
        pass
    # Checkpoint key layouts differ from the module tree transformers exposes, and a
    # multimodal checkpoint puts the text stack under `language_model` beside a visual tower.
    # Named candidates, tried in order -- never a search for anything norm-shaped.
    NORMS = ("model.language_model.norm.weight", "model.norm.weight")
    HEADS = ("lm_head.weight", "model.language_model.embed_tokens.weight",
             "model.embed_tokens.weight")
    wanted = NORMS + HEADS
    got = {}
    files = sorted({idx[w] for w in wanted if idx and w in idx}) if idx else ["model.safetensors"]
    for f in files:
        t = load_file(hf_hub_download(repo, f))
        for w in wanted:
            if w in t:
                got[w] = t[w].to(torch.float32)
    norm = next((got[k] for k in NORMS if k in got), None)
    head = next((got[k] for k in HEADS if k in got), None)
    if norm is None or head is None:
        raise SystemExit(f"refusing: {repo} matched no known readout layout "
                         f"(norm={norm is not None}, head={head is not None}). Add the layout "
                         f"rather than letting the run guess which tensor is the readout.")
    return norm, head


def rms_unembed(norm_w, head_w, keep):
    """Qwen's readout on a token subset: RMSNorm with a learned gain, then the head."""
    import torch
    Wk = head_w[torch.as_tensor(np.asarray(keep))]

    def _un(X):
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w
        return (t @ Wk.T).to(torch.float64).numpy()
    return _un


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="the model run in-process")
    ap.add_argument("--b", required=True, help="the model whose streams are on disk")
    ap.add_argument("--lens-a", required=True, type=Path)
    ap.add_argument("--lens-b", required=True, type=Path)
    ap.add_argument("--streams-b", required=True, type=Path)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--keep", type=int, default=4096, help="size of the shared token sub-basis")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)

    la, lb = je.load_lens(a.lens_a), je.load_lens(a.lens_b)
    tok_a = transformers.AutoTokenizer.from_pretrained(a.a)
    tok_b = transformers.AutoTokenizer.from_pretrained(a.b)
    va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    if len(va) != len(vb):
        raise SystemExit(f"refusing: vocabularies differ in size ({len(va)} vs {len(vb)}); the "
                         f"two sides would not be landing on one basis")
    ia = {i: s for s, i in va.items()}
    ib = {i: s for s, i in vb.items()}
    n_same = sum(1 for i in range(len(va)) if ia.get(i) == ib.get(i))
    if n_same != len(va):
        raise SystemExit(f"refusing: only {n_same}/{len(va)} token ids agree between the two "
                         f"tokenizers. A screen needs ONE basis; coupling across two different "
                         f"ones reports magnitude alone.")
    print(f"shared basis: {n_same} token ids identical across both tokenizers")

    if not a.streams_b.is_file():
        raise SystemExit(f"refusing: {a.streams_b} does not exist")
    z = np.load(a.streams_b)
    Sb = [z[k].astype(np.float64) for k in sorted((k for k in z.files if k.startswith("s")),
                                                  key=lambda k: int(k[1:]))]
    logits_b = z["logits"].astype(np.float64)

    model = transformers.AutoModelForCausalLM.from_pretrained(a.a).eval()
    if next(model.parameters()).dtype != torch.float32:
        model = model.float()
    ids = wikitext_prompts(tok_a, a.prompts, a.tokens)
    Sa, logits_a = collect_streams(model, ids)
    if len(Sa) != len(Sb) or Sa[0].shape[1] != Sb[0].shape[1]:
        raise SystemExit(f"refusing: {len(Sa)}x{Sa[0].shape[1]} against "
                         f"{len(Sb)}x{Sb[0].shape[1]} -- the two sides did not see the same "
                         f"text, so a row-paired coupling would compare different positions")
    print(f"A {a.a}: d={Sa[0].shape[-1]}, {len(la)} fitted layers")
    print(f"B {a.b}: d={Sb[0].shape[-1]}, {len(lb)} fitted layers  (streams from disk)")

    # The shared token sub-basis: where BOTH models put their mass. Taken from each model's own
    # final-layer output and unioned, so neither side chooses the coordinates alone.
    def mass(L):
        e = np.exp(L - L.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).sum(0)
    # Ranked by the probability mass each side actually puts on a token, summed over positions,
    # then averaged so neither side alone fixes the coordinates. A per-position top-k union
    # returns tens of thousands of ids and is not a sub-basis.
    score = 0.5 * (mass(logits_a) / len(logits_a) + mass(logits_b) / len(logits_b))
    keep = np.sort(np.argpartition(-score, a.keep)[:a.keep])
    covered = float(score[keep].sum() / score.sum())
    print(f"shared token sub-basis: {keep.size} ids carrying {covered:.1%} of both sides' "
          f"mean output mass")

    un_a = unembed_fn(model)
    norm_b, head_b = load_readout(a.b)
    un_b_full = rms_unembed(norm_b, head_b, keep)

    def side_a(H):
        return un_a(np.asarray(H))[:, keep]

    depth_a, depth_b = la.source_layers[-1] or 1, lb.source_layers[-1] or 1
    rows = []
    print(f"\n{'rel':>5} {'L_A':>4} {'L_B':>4} {'couple':>9} {'z':>9} {'strength':>9} "
          f"{'tight':>7} {'sign':>5}")
    for la_i in la.source_layers:
        rel = la_i / depth_a
        lb_i = min(lb.source_layers, key=lambda l: abs(l / depth_b - rel))
        Ja, Jb = la.jacobian(la_i), lb.jacobian(lb_i)
        Xa = np.concatenate([side_a(s[la_i + a.offset] @ Ja.T) for s in Sa], axis=0)
        Xb = np.concatenate([un_b_full(s[lb_i + a.offset] @ Jb.T) for s in Sb], axis=0)
        s_ = Screen(far=a.far)
        s_.register(A, entry=lambda X: np.asarray(X, dtype=np.float64))
        s_.register(B, entry=lambda X: np.asarray(X, dtype=np.float64))
        s_.place(A, Xa)
        s_.place(B, Xb)
        c = s_.coupling(A, B)
        row = {"rel_depth": rel, "layer_a": la_i, "layer_b": lb_i,
               "couple": float(s_.couple(A, B)), "z": float(c.z),
               "strength": float(c.strength), "sign": float(c.sign),
               "tight": float(c.tightness), "keep": int(keep.size)}
        rows.append(row)
        print(f"{rel:>5.2f} {la_i:>4} {lb_i:>4} {row['couple']:>9.4f} {row['z']:>9.2f} "
              f"{row['strength']:>9.4f} {row['tight']:>7.3f} {row['sign']:>5.0f}", flush=True)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(
                {"a": a.a, "b": a.b, "lens_a": str(a.lens_a), "lens_b": str(a.lens_b),
                 "prompts": len(Sa), "tokens": a.tokens, "keep": int(keep.size),
                 "far": a.far, "offset": a.offset, "layers": rows}, indent=2))
    print(f"\nwrote {a.out}" if a.out else "")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
