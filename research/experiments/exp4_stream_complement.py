"""Experiment 4 -- what stands outside J-space, measured on real residual streams.

This is the number the workspace argument rests on and does not report. [Anthropic 2026] gives
J-space as 6-10% of activation variance and treats the remainder as automatic processing. A
variance share is silent about structure: 94% of the variance could be a hundred resolved
directions or none, and those support opposite readings of whether the workspace is the model's
whole flexible capacity.

Unlike experiments 1-3, a detection floor is legitimate here. A residual stream is ONE sample;
a transport is a corpus average whose noise the averaging already removed. The Tracy-Widom edge
applies to the stream as it does to any 2-D field with an ordered axis (positions) and a feature
axis (residual dimensions).

Reads, per layer, on the screen of ``lenses.layer_screen``:

  certify        the round trip through the transport and back. By Prop. 2.2 its residual is the
                 component of the stream in ker(J), and its K_signal says whether that component
                 carries structure the instrument can see.
  complement     ``uncondensed(stream, jlens).K_signal`` -- modes of the stream standing above
                 the floor OUTSIDE what the transport resolves.
  transfer       the energy accounting: participation, tau, bystanding.

Layer alignment
---------------
``ActivationRecorder`` hooks the residual **blocks** and captures their OUTPUT, and
``LensModel.layers`` is that block ModuleList indexed ``0..n_layers-1``. So lens layer ``l`` is
the output of block ``l``, which is ``hidden_states[l+1]`` -- ``hidden_states[0]`` being the
embeddings. That is the convention, taken from the reference implementation rather than inferred.

It is still checked, in logit space, because an off-by-one would silently pair every stream with
the wrong transport. Top-1 agreement at positions the model is confident about is the
discriminating statistic; a global cosine is not, because adjacent streams are strongly
correlated and ``J`` is close to ``alpha I``, so a shifted pairing scores well on cosine while
predicting the wrong token. Letting cosine decide is exactly how an earlier version of this
script produced a full table of numbers built on the wrong pairs.

Usage
-----
    python experiments/exp4_stream_complement.py --model gpt2 \\
        --lens lenses/gpt2-small/.../gpt2_jacobian_lens.pt \\
        --prompts 8 --tokens 128 --out results/streams.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402
from entroptics import Projection                                      # noqa: E402


def wikitext_prompts(tok, n: int, tokens: int, skip: int = 0) -> list:
    """``n`` sequences of exactly ``tokens`` tokens from wikitext-2 -- the corpus family the
    published lenses were fitted on, so the streams match the fit distribution.

    ``skip`` discards the first ``skip`` sequences, which is how a *second sample* is drawn. Every
    run took the same leading sequences until this existed, so rerunning an experiment reproduced
    its prompts exactly and could never test whether a number survived a different draw. It does
    not always: sec 7.1's claim that gemma-3-1b's complement removes exactly the transport's
    carried rank held on the first eight sequences and gave 9.9 against 12.0 on another eight,
    because the removal has a per-prompt standard deviation of 5.7.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out, buf = [], []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200:
            continue
        buf.append(t)
        ids = tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
        if ids.shape[0] >= tokens:
            out.append(ids[:tokens])
            buf = []
            if len(out) >= n + skip:
                break
    out = out[skip:]
    if len(out) < n:
        raise SystemExit(f"refusing: wikitext yielded {len(out)} sequences of {tokens} tokens, "
                         f"asked for {n}. Lower --tokens or --prompts rather than padding: a "
                         f"padded sequence is not {tokens} tokens of signal.")
    return out


def collect_streams(model, ids_list):
    """``hidden_states`` and the model's own logits, per prompt.

    Returns ``(streams, logits)``: a list of ``(n_hidden, T, d)`` arrays and the concatenated
    ``(N*T, vocab)`` logits the model actually produced, which the alignment check scores against.
    """
    import torch
    streams, logits = [], []
    with torch.no_grad():
        for ids in ids_list:
            out = model(ids.unsqueeze(0).to(model.device), output_hidden_states=True)
            streams.append(np.stack([h[0].to(torch.float64).cpu().numpy()
                                     for h in out.hidden_states]))
            logits.append(out.logits[0].to(torch.float64).cpu().numpy())
    return streams, np.concatenate(logits, axis=0)


def unembed_fn(model):
    """The model's own readout: final layer norm, then the output head.

    This is what ``jlens`` means by ``model.unembed``, and composing with it is what makes the
    transport interpretable at all -- see ``check_alignment``.
    """
    import torch

    # Named layouts, tried in order. Not a search for anything norm-shaped: picking the wrong
    # module would silently change what every number here measures, so an unknown architecture
    # refuses and asks to be added rather than being guessed at.
    LAYOUTS = (
        ("gpt2",     "transformer.ln_f",           "lm_head"),
        ("gptneox",  "gpt_neox.final_layer_norm",  "embed_out"),
        ("llama-ish", "model.norm",                "lm_head"),   # llama, qwen, mistral, olmo
        ("gemma3",   "model.language_model.norm",  "lm_head"),
    )

    def reach(root, path):
        cur = root
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur

    for name, npath, hpath in LAYOUTS:
        norm, head = reach(model, npath), reach(model, hpath)
        if norm is not None and head is not None:
            print(f"  readout: {name} ({npath} + {hpath})")
            break
    else:
        raise SystemExit(
            f"refusing: no known readout layout matches {type(model).__name__}. The transport is "
            f"only meaningful composed with the model's own final norm and output head "
            f"(jlens: 'final norm + LM head'); add this architecture to LAYOUTS rather than "
            f"letting the run guess which module to use.")

    dev = next(head.parameters()).device

    def _un(A):
        t = torch.tensor(np.asarray(A), dtype=torch.float32, device=dev)
        return head(norm(t)).to(torch.float64).cpu().numpy()
    return _un


def check_alignment(streams, lens, unembed, model_logits, offsets=(0, 1, 2)) -> int:
    """Which ``hidden_states`` index does lens layer ``l`` refer to?

    **Scored in logit space, not residual space.** The transport is only meaningful composed
    with the unembedding: ``jlens`` feeds the raw residual to ``model.unembed``, which for a
    HuggingFace decoder is the final layer norm followed by the output head. In the raw residual
    basis ``H_l J_l.T`` approximates nothing -- measured on gpt2 it correlates ~0.0 with the
    final stream and about -0.65 with its OWN input, and its norm runs to 6x the target, because
    a Jacobian maps perturbations and ``ln_f`` discards exactly the scale and mean that dominate
    an absolute activation. Composed with the unembedding the same transport agrees with the
    model's own logits from 0.13 at layer 0 to 0.86 at the last fitted layer, with top-1 token
    agreement reaching 0.92.

    Two statistics per offset, and they must agree: centred cosine over the vocabulary axis, and
    the fraction of positions whose top-1 token matches. The second is what catches a degenerate
    pairing that the first likes -- feeding the already-normalised final stream back through
    ``ln_f`` scores well on cosine while its top-1 agreement collapses.
    """
    n_hidden = streams[0].shape[0]
    ml = model_logits - model_logits.mean(1, keepdims=True)
    ml_n = np.sqrt((ml ** 2).sum(1))
    best_tok = model_logits.argmax(1)
    # Top-1 agreement only discriminates where the model itself is decided. On natural text most
    # positions are near-uniform over plausible continuations, and every offset then scores the
    # same ~0.2 -- which reads as "no discrimination" and would let a thin cosine margin choose.
    e = np.exp(model_logits - model_logits.max(1, keepdims=True))
    conf = (e / e.sum(1, keepdims=True)).max(1) >= 0.5
    if conf.sum() < 16:
        raise SystemExit(f"refusing: only {int(conf.sum())} of {conf.size} positions have the "
                         f"model above p=0.5, too few to settle the layer pairing. Use more or "
                         f"more predictable prompts rather than deciding on cosine alone.")
    # One upcast per layer, not one per offset: J is d x d in float16 on disk and
    # every read needs float64, so re-reading it inside the offset loop converts
    # 6.5M elements three times over at d=2560.
    jac = {l: lens.jacobian(l) for l in lens.source_layers}
    scores, tops = {}, {}
    for off in offsets:
        cos_v, top_v = [], []
        for l in lens.source_layers:
            i = l + off
            if not 0 <= i < n_hidden:
                cos_v = []
                break
            H = np.concatenate([s[i] for s in streams], axis=0)
            L = unembed(H @ jac[l].T)
            a = L - L.mean(1, keepdims=True)
            den = np.sqrt((a ** 2).sum(1)) * ml_n
            cos_v.append(float(np.mean((a * ml).sum(1) / np.maximum(den, 1e-30))))
            top_v.append(float((L.argmax(1)[conf] == best_tok[conf]).mean()))
        if cos_v:
            scores[off], tops[off] = float(np.mean(cos_v)), float(top_v[-1])
    if not scores:
        raise SystemExit("refusing: no candidate offset maps every fitted layer into range")

    print(f"  alignment, in LOGIT space over {int(conf.sum())} confident positions "
          f"(p>=0.5) of {conf.size}:")
    for off in sorted(scores):
        print(f"    hidden_states[l{off:+d}]  cos={scores[off]:+.4f}  top1={tops[off]:.2f}")
    return scores, tops


def verify_alignment(offset, scores, tops) -> None:
    """Corroborate the documented pairing; do not re-derive it from statistics.

    The convention is settled by the reference implementation, not by a correlation:
    ``ActivationRecorder`` hooks the residual **blocks** and captures their OUTPUT, and
    ``LensModel.layers`` is the block ModuleList indexed 0..n_layers-1. So lens layer ``l`` is
    the output of block ``l``, which is ``hidden_states[l+1]`` in HuggingFace's convention --
    ``hidden_states[0]`` being the embeddings.

    The statistics are kept as a check on that reading. Top-1 agreement is the discriminating
    one: adjacent residual streams are strongly correlated and ``J`` is close to ``alpha I``, so
    a pairing shifted by one block still scores well on a global cosine while predicting the
    wrong token. Measured on gpt2, offset +1 wins top-1 (0.57 against 0.46 and 0.35) while
    cosine mildly prefers +2 -- which is why the earlier version of this check, which let cosine
    decide, produced a full table of numbers on the wrong pairs.

    Refuses only if the documented offset is worst on BOTH statistics, which would mean the
    convention does not hold for this model and the run should stop.
    """
    if offset not in scores:
        raise SystemExit(f"refusing: offset {offset:+d} was not scored; candidates were "
                         f"{sorted(scores)}")
    worst_cos = offset == min(scores, key=scores.get)
    worst_top = offset == min(tops, key=tops.get)
    best_top = max(tops, key=tops.get)
    if worst_cos and worst_top and len(scores) > 1:
        raise SystemExit(
            f"refusing: hidden_states[l{offset:+d}] is the documented pairing but scores worst "
            f"on both statistics here (cos={scores[offset]:+.4f}, top1={tops[offset]:.2f}). The "
            f"convention does not hold for this model; settle it before reading anything.")
    note = "corroborated" if offset == best_top else f"NOTE top-1 prefers l{best_top:+d}"
    print(f"  -> using hidden_states[l{offset:+d}] (documented: blocks are hooked at their "
          f"output) -- {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a HuggingFace decoder, e.g. gpt2")
    ap.add_argument("--lens", required=True, type=Path)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--skip", type=int, default=0,
                    help="discard this many leading sequences, to draw a SECOND sample. "
                         "Every run took the same leading prompts until this existed, so "
                         "rerunning could never test whether a number survives a "
                         "different draw -- and sec 7.1's did not.")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--dump-streams", type=Path, default=None,
                    help="run only the forward passes and save streams+logits to this .npz, "
                         "then exit. For splitting a run across machines.")
    ap.add_argument("--streams-from", type=Path, default=None,
                    help="load streams+logits from a .npz instead of running the model")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; the forward pass moves, every spectral read stays on CPU in float64")
    ap.add_argument("--realise", action="store_true",
                    help="also measure what the crossing actually delivers through the "
                         "receiver's own inverse (an extra round trip per read)")
    ap.add_argument("--offset", type=int, default=1,
                    help="hidden_states index for lens layer l (default 1: blocks "
                         "are hooked at their output, so l -> hidden_states[l+1])")
    ap.add_argument("--rank", default="pr",
                    help="transport rank: 'pr' (participation ratio of J-aI, "
                         "threshold-free), 'full', 'resolved', or an int")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import torch
    import transformers
    torch.set_grad_enabled(False)

    lens = je.load_lens(a.lens)

    # Forward passes and spectral reads want different machines. A GPU box does the former
    # fast and may do the latter very slowly -- measured on one pod, an SVD at d=1024 takes
    # 1.5s single-threaded and 25.6s at 16 OpenBLAS threads, against 0.23s on the desktop.
    # So the two phases split: --dump-streams runs the model and stops, --streams-from skips
    # the model entirely.
    if a.streams_from is not None:
        if not a.streams_from.is_file():
            raise SystemExit(f"refusing: {a.streams_from} does not exist")
        z = np.load(a.streams_from)
        streams = [z[k].astype(np.float64)
                   for k in sorted((k for k in z.files if k.startswith("s")),
                                   key=lambda k: int(k[1:]))]
        model_logits = z["logits"].astype(np.float64)
        d = int(streams[0].shape[-1])
        if d != lens.d_model:
            raise SystemExit(f"refusing: dumped streams have d={d} but the lens has "
                             f"d={lens.d_model} -- these are different models")
        meta_dtype = str(z["dtype_source"]) if "dtype_source" in z.files else "unknown"
        print(f"streams from {a.streams_from}: {len(streams)} prompts, "
              f"{streams[0].shape[0]} hidden states of {streams[0].shape[1:]}, "
              f"source dtype {meta_dtype}")
        model = tok = None
        ids_list = None      # no prompts were drawn; the
                             # fingerprint below records that honestly
        src_dtype = meta_dtype
    else:
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(a.model).eval()
        src_dtype = next(model.parameters()).dtype
        if src_dtype != torch.float32:
            # bfloat16 carries ~3 decimal digits. Every read here is spectral, and passing dtype=
            # to from_pretrained does not convert a tied-embedding bf16 checkpoint (measured on
            # Qwen3.5-0.8B: lm_head stayed bf16 and the readout raised on the mismatch).
            model = model.float()
            print(f"  cast {src_dtype} -> float32 for the forward pass "
                  f"(storage precision is unchanged; the arithmetic is not)")
        if a.device != "cpu":
            model = model.to(a.device)
            print(f"  model on {a.device}: {torch.cuda.get_device_name(0)}"
                  if a.device.startswith("cuda") else f"  model on {a.device}")
        d = model.config.hidden_size
        if d != lens.d_model:
            raise SystemExit(f"refusing: model d_model={d} but lens d_model={lens.d_model} -- this "
                             f"lens was not fitted on this model")

        print(f"model {a.model}  d={d}  lens {a.lens.name}  n_prompts={lens.n_prompts}")
        ids_list = wikitext_prompts(tok, a.prompts, a.tokens, skip=a.skip)
        print(f"collected {len(ids_list)} x {a.tokens}-token wikitext sequences", flush=True)
        streams, model_logits = collect_streams(model, ids_list)
        print(f"hidden_states: {streams[0].shape[0]} tensors of {streams[0].shape[1:]} per prompt")

        if a.dump_streams is not None:
            # Save and stop. The spectral half runs elsewhere, on whatever machine has
            # the faster LAPACK -- the numbers are identical either way.
            a.dump_streams.parent.mkdir(parents=True, exist_ok=True)
            # Stored float32: the forward pass ran in float32, so float64 on disk doubles
            # the bytes for no information. Reads upcast on load, where it matters.
            np.savez_compressed(
                a.dump_streams, logits=model_logits.astype(np.float32),
                dtype_source=np.array(str(src_dtype)),
                **{f"s{k}": v.astype(np.float32) for k, v in enumerate(streams)})
            mb = a.dump_streams.stat().st_size / 1e6
            print(f"wrote {a.dump_streams} ({mb:.0f} MB); rerun with --streams-from")
            return 0

    if model is not None:
        scores, tops = check_alignment(streams, lens, unembed_fn(model), model_logits)
        verify_alignment(a.offset, scores, tops)
    else:
        # No model in this process, so no logit-space check is possible. The documented
        # pairing still applies; the record must not imply it was corroborated here.
        scores, tops = {}, {}
        print(f"  alignment: not checked in this process (streams loaded from disk); "
              f"using the documented hidden_states[l{a.offset:+d}]")
    off = a.offset
    layers = (lens.source_layers if a.layers.strip() == "all"
              else [int(v) for v in a.layers.split(",") if v])

    # A fingerprint of the token ids actually drawn. `--skip` was added to make a second
    # sample possible and its first use silently drew the SAME prompts, because the flag reached
    # the provenance block without reaching the call. The provenance said `prompt_skip: 8` and the
    # numbers were identical to three decimals. A sample is only a different sample if the ids
    # differ, so the ids are hashed and recorded rather than the intent.
    import hashlib as _hl
    prompt_fp = (_hl.sha256(b"".join(bytes(memoryview(i.numpy())) for i in ids_list)).hexdigest()[:16]
                 if ids_list is not None else None)

    def write_out(complete: bool) -> None:
        # Provenance, to the same standard as exp1: every section 7 claim comes out of this
        # file, so the record has to say which artefact and which arithmetic produced it. The
        # cast matters -- a bf16 forward pass and a float32 one are not the same measurement.
        #
        # The completion stamp originated here and is now `je.dump` / `je.load_complete`, shared
        # by every experiment that writes incrementally -- see `entroptics_jlens.results`.
        import hashlib
        import platform
        h = hashlib.sha256()
        with open(a.lens, "rb") as fh:
            h.update(fh.read(1 << 24))
        je.dump(a.out,
            {"model": a.model, "lens": str(a.lens), "lens_sha256_head": h.hexdigest(),
             "lens_n_prompts": lens.n_prompts, "d_model": d, "offset": off,
             "offset_source": "documented (blocks hooked at output)",
             "alignment_scores": {str(k): v for k, v in scores.items()},
             "alignment_top1": {str(k): v for k, v in tops.items()},
             "prompts": len(streams), "tokens": a.tokens, "far": a.far,
             "prompt_skip": a.skip, "prompt_fingerprint": prompt_fp,
             "rank_specs": specs, "corpus": "Salesforce/wikitext wikitext-2-raw-v1 test",
             "model_dtype_source": str(src_dtype), "model_dtype_used": "torch.float32",
             "device": a.device,
             "environment": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__,
                             "transformers": transformers.__version__,
                             "platform": platform.platform()},
             "layers": rows},
            complete=complete)

    print(f"\n{'layer':>5} {'rankspec':>9} {'rank':>6} {'Tdirs':>7} {'K(str)':>7} {'K(cmp)':>7} {'out%':>7} {'resid':>7} {'Jshare':>7} {'modes_to':>9} {'match':>7} {'tau':>8}", flush=True)
    rows = []
    specs = [v.strip() for v in a.rank.split(",") if v.strip()]
    for l in layers:
        i = l + off
        J = lens.jacobian(l)
        for spec in specs:
            # The transport's rank. `full` leaves J at full rank, where its pseudo-inverse is
            # ill-conditioned: the round trip returns a residual of order 1e-5 that carries no
            # information, and the screen's per-channel whitening lifts that round-off to unit
            # amplitude and resolves "modes" in it. certify needs a genuine null space, and the
            # participation ratio supplies one with no floor involved.
            #
            # Sweeping several is not decoration. `outside%` is bounded by the rank -- a rank-2
            # transport occupies at most 2 directions, so the complement keeps ~everything by
            # arithmetic. Reading one rank and reporting the fraction would dress that identity
            # up as a finding.
            if spec == "pr":
                rank = max(1, int(round(je.participation_ratio(
                    je.energy_spectrum(je.decompose(J, kind="identity").residual)))))
            elif spec in ("full", "resolved"):
                rank = None if spec == "full" else "resolved"
            else:
                rank = min(int(spec), lens.d_model)
            # Truncate ONCE per (layer, rank), not once per prompt. The truncation depends
            # only on J and the rank, never on the stream, and at d=2560 each one is a full
            # SVD -- doing it inside the prompt loop was four redundant 2560x2560 SVDs per
            # layer and made the 4B run slower than every CPU run combined. The truncated
            # transport is then passed with rank=None, which uses the matrix as given.
            # The truncation and its inverse come from ONE SVD, as a matched pair. Calling
            # np.linalg.pinv on the reconstructed truncation is the obvious move and it is
            # wrong: at gpt2 layer 9, rank 3, it gave a round trip with top singular value
            # 1.31 where an orthogonal projector has exactly 1, and certify then read a
            # residual above 1 -- an energy "share" of -7.8%. See je.truncated_pair.
            if rank is None:
                J_used = J
                J_pinv = np.linalg.pinv(je.as_frame(J_used))
            elif rank == "resolved":
                J_used, spec_ = je.resolved_transport(J, far=a.far)
                J_used, J_pinv = je.truncated_pair(J, spec_.K)
            else:
                J_used, J_pinv = je.truncated_pair(J, rank)
            J_spec = je.transport_spectrum(J_used, far=a.far)
            per = []
            for s in streams:
                H = s[i]
                ls_ = je.layer_screen(H, J_used, layer=l, far=a.far, rank=None,
                                      pinv=J_pinv, spectrum=J_spec)
                sc = ls_.screen
                stream_K = Projection(H, far=a.far).K_signal
                comp = je.complement(ls_)
                cert = sc.certify(je.TRANSPORT, H)
                tr = sc.transfer(je.STREAM, je.TRANSPORT)
                rec = {"K_stream": stream_K, "K_complement": comp.K_signal,
                       "rank": (rank if isinstance(rank, int)
                                else (J_used.shape[0] if rank is None
                                      else int(np.linalg.matrix_rank(
                                          J_used, tol=1e-8 * np.linalg.norm(
                                          J_used, 2))))),
                       "K_transport_dirs": int(sc.directions(je.TRANSPORT).shape[1]),
                       "certify_K": cert.K_signal, "certify_residual": cert.residual,
                       "participation": float(tr.participation), "tau": float(tr.tau),
                       "bystanding": float(tr.bystanding),
                       # The etendue accounting is already computed inside Transfer, so these
                       # cost nothing extra. `modes_to` is the capacity BOUND: how many
                       # directions the receiving side's phase space can carry, independent of
                       # how many are observed occupied. That is the quantity the workspace
                       # capacity figure is an observation of, not a bound on.
                       "etendue_from": float(tr.etendue_from),
                       "etendue_to": float(tr.etendue_to),
                       "etendue_match": float(tr.match),
                       "modes_from": int(tr.modes_from), "modes_to": int(tr.modes_to),
                       "concentration": float(tr.concentration)}
                if a.realise:
                    r = sc.realise(je.STREAM, je.TRANSPORT)
                    rec.update({"realise_efficiency": float(r.efficiency),
                                "realise_shortfall": float(r.shortfall),
                                "realise_passive": bool(r.passive)})
                per.append(rec)
            agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            outside = agg["K_complement"] / agg["K_stream"] if agg["K_stream"] else float("nan")
            row = {"layer": l, "hidden_index": i, "rank_spec": spec, "n_prompts": len(per),
                   "outside_fraction": outside, **agg, "per_prompt": per}
            rows.append(row)
            # Stream energy inside J-space. Prop 2.2 makes this a share only while the
            # round trip contracts: a residual above 1 means it AMPLIFIED, and
            # 1 - residual^2 is then a negative "share of energy", which is no
            # measurement at all. Measured at gpt2 layer 9, where the transport falls to
            # rank 3 of 768: residual 1.038, share -7.8%. Report it as undefined rather
            # than print a number outside [0, 1] in a column of shares.
            res = agg["certify_residual"]
            jshare = 1.0 - res ** 2 if res <= 1.0 else float("nan")
            row["j_space_energy_share"] = jshare
            row["certify_amplified"] = bool(res > 1.0)
            # Formatted outside the f-string: reusing the outer quote character inside one is a
            # syntax error before Python 3.12, and this package supports 3.10.
            jshare_cell = "  n/a" if jshare != jshare else format(jshare, ">7.1%")
            print(f"{l:>5} {spec:>9} {agg['rank']:>6.0f} {agg['K_transport_dirs']:>7.1f} "
                  f"{agg['K_stream']:>7.1f} {agg['K_complement']:>7.1f} {outside:>7.1%} "
                  f"{agg['certify_residual']:>7.4f} {jshare_cell:>7} {agg['modes_to']:>9.1f} "
                  f"{agg['etendue_match']:>7.3f} {agg['tau']:>8.4f}", flush=True)

            # Written after EVERY layer, not at the end. A run killed at layer 19 of 27
            # otherwise loses forty minutes of compute and leaves no file at all.
            if a.out:
                write_out(False)

    if a.out:
        write_out(True)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.LensFormatError, ValueError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
