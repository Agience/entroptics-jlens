"""Does an injected ontology entry make the answer BETTER, or only different?

This was the one load-bearing unknown in a since-removed design note. Everything measured so far
says the write path STEERS: a direction resolved in one system, rendered into another's basis, moves
the receiver toward the tokens the sender named -- against controls, held out, with a diagonal
specificity matrix, and (2026-08-22aw) twenty such directions compose without destroying each other.
None of that shows an answer got better. The one attempt at benefit was +1.7% on development and
-1.2% held out.

So: a task with a right answer, and the entry is a real ontology entry.

    task        reverse dictionary. Given a gloss, name the word. This is genesis's own bench
                shape -- `bench_reverse_dictionary`, 89,614 corpus-labelled questions -- and the
                answer is unambiguous: a lemma of the synset the gloss belongs to.
    score       two readings of one forward pass. The RANK of the correct lemma's token among the
                model's whole vocabulary, and a K-way FORCED CHOICE against distractor lemmas drawn
                from the same pool -- which is what genesis's own bench reports, and which has far
                more power at this sample size.

The prompt is few-shot, and that was settled before any arm was measured. Four formats on 30 items:
plain `The word that means "..." is` reads rank-1 0.0% / rank-10 0.0% / median rank 3,994, and
three few-shot `Definition: ... / Word: ...` pairs read rank-1 33.3% / rank-10 76.7% / median 3.
A benchmark whose baseline cannot reach rank 10 measures nothing, so the first run of this script
refused itself on exactly that guard. The shots come from a pool disjoint from both splits.

THE ENTRY IS crystal's OWN GEOMETRY. `crystal/ontology/geometry.py` builds a synset's coordinate
as `vec_sparse(s)`: the root->s hypernym path, each edge parent p -> child c weighted by
`sqrt(IC(c) - IC(p))`. That vector IS the ontology's statement about where the concept sits. Here it
is rendered onto the model's vocabulary -- each ancestor's weight placed on the tokens that spell
its lemmas -- and crossed into the residual basis by the inverse readout. So what gets injected is
not a summary someone chose; it is the same object the retrieval geometry already runs on.

The target's own lemmas are removed from the profile. Without that the injection carries the
answer and the test measures nothing.

Five arms, and the fifth is the one that decides whether any of this is worth building:

    base          the prompt alone
    inject        the prompt plus the entry, injected as a direction
    shuffled      the same energy, vocabulary permuted -- content-free, same crossing
    wrong         a DIFFERENT synset's entry -- same construction, wrong concept
    prompt_text   the same ancestors appended to the prompt AS WORDS

`prompt_text` is the dumbest thing that could work, and nothing in this project has run it. If
pasting the ontology entry into the prompt beats injecting it, the write path is decoration. Note
what a tie would mean: injection costs no context tokens and composes (n=20 at 55% retention),
where n entries of prompt text cost n times the context. Parity is not nothing; losing is fatal.

Nothing is tuned on the reported set. Layer and amplitude are free parameters, so they are chosen
on a DEVELOPMENT split and the arms are reported on a disjoint TEST split. The development surface
is printed in full. This is the discipline the handoff claim failed: alpha = 1.0 was a development
argmax that read -1.2% held out.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import entroptics_jlens as je                                          # noqa: E402
from exp26_cross_tokenizer import blocks_of, head_of                   # noqa: E402

SHOT = "Definition: {g}\nWord: {w}\n\n"
ASK = "Definition: {g}\nWord:"
ASK_TEXT = "Definition: {g}; a kind of {a}\nWord:"


def ontology_entry(syn, ic, drop: set):
    """crystal's `vec_sparse`: the root->s hypernym path, edge p->c weighted sqrt(IC(c)-IC(p)).

    Returned as {lemma string: weight} over the ANCESTORS only. The synset's own lemmas are in
    `drop` -- carrying them would inject the answer.
    """
    from nltk.corpus.reader.wordnet import information_content

    def icv(s):
        try:
            return float(information_content(s, ic))
        except Exception:
            return 0.0

    out = {}
    for path in syn.hypernym_paths():
        for parent, child in zip(path, path[1:]):
            if child == syn:
                continue
            d = icv(child) - icv(parent)
            if d <= 0.0:
                continue
            w = math.sqrt(d)
            for lem in child.lemmas():
                name = lem.name().replace("_", " ")
                if name.lower() in drop:
                    continue
                out[name] = max(out.get(name, 0.0), w)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dev", type=int, default=40)
    ap.add_argument("--test", type=int, default=120)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--choices", type=int, default=20,
                    help="size of the forced-choice candidate set, answer included")
    ap.add_argument("--layers", default="3,8,13")
    ap.add_argument("--amps", default="0.25,0.5")
    ap.add_argument("--where", default="prefix,span,final",
                    help="injection site. prefix = the first token of the question, which "
                         "every later token attends to; span = the whole question; final = "
                         "the answer position, which nothing attends to")
    ap.add_argument("--pad", type=int, default=2048,
                    help="random vocabulary rows added to the crossing surface, so the inverse "
                         "is not solved on the ontology's own tokens alone")
    ap.add_argument("--out", type=Path, default=Path("results/benefit.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from nltk.corpus import wordnet as wn
    from nltk.corpus import wordnet_ic
    torch.set_grad_enabled(False)

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    M = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32).eval().float()
    W = head_of(M)
    blocks = blocks_of(M)
    d = W.shape[1]
    ic = wordnet_ic.ic("ic-brown.dat")
    print(f"{a.model}: d={d}, {len(blocks)} blocks, vocab {W.shape[0]:,}", flush=True)

    def one_token(s: str):
        ids = tok(" " + s, add_special_tokens=False).input_ids
        return ids[0] if len(ids) == 1 else None

    rng = np.random.default_rng(0)
    # `hypernym_paths()` walks the DAG; running it over all 82,115 nouns to filter costs
    # far more than it saves, since the loop below computes the path for kept items anyway.
    nouns = [s for s in wn.all_synsets("n") if s.definition()]
    order = rng.permutation(len(nouns))
    items = []
    need = a.shots + a.dev + a.test
    for oi in order:
        syn = nouns[int(oi)]
        own = {l.name().replace("_", " ").lower() for l in syn.lemmas()}
        answer = None
        for lem in syn.lemmas():
            w = lem.name().replace("_", " ")
            if w.isalpha() and one_token(w) is not None:
                answer, ans_id = w, one_token(w)
                break
        if answer is None:
            continue
        entry = ontology_entry(syn, ic, own)
        prof = {}
        for name, weight in entry.items():
            t = one_token(name)
            if t is not None and t != ans_id:
                prof[t] = max(prof.get(t, 0.0), weight)
        if len(prof) < 3:                       # an entry with nothing renderable is not a test
            continue
        items.append({"syn": syn.name(), "gloss": syn.definition(), "answer": answer,
                      "ans_id": ans_id, "profile": prof,
                      "words": [w for w, _ in sorted(entry.items(), key=lambda kv: -kv[1])[:3]]})
        if len(items) >= need:
            break
    if len(items) < need:
        raise ValueError(f"only {len(items)} usable items, {need} required -- refusing to "
                         f"report a smaller sample than asked for")
    shots, dev, test = (items[:a.shots], items[a.shots:a.shots + a.dev],
                        items[a.shots + a.dev:])
    prefix = "".join(SHOT.format(g=it["gloss"], w=it["answer"]) for it in shots)
    print(f"{len(shots)} shots, {len(dev)} development items, {len(test)} test items", flush=True)

    # Forced-choice distractors: other items' answers, fixed per item so every arm faces the
    # same question. Drawn from the same single-token noun pool as the answer, so the choice is
    # between words of the same kind rather than between a word and an artefact of tokenisation.
    pool = [it["ans_id"] for it in items]
    if len(set(pool)) < a.choices:
        raise ValueError(f"{len(set(pool))} distinct answers cannot fill a "
                         f"{a.choices}-way choice: raise --dev/--test or lower --choices")
    for k, it in enumerate(items):
        r = np.random.default_rng(1000 + k)
        cand = {it["ans_id"]}
        while len(cand) < a.choices:
            cand.add(int(pool[int(r.integers(len(pool)))]))
        it["choices"] = np.array(sorted(cand), dtype=int)
    print(f"  e.g. {test[0]['syn']}: answer '{test[0]['answer']}', "
          f"entry names {test[0]['words']}", flush=True)

    # The crossing surface: every token any entry lands on, plus random vocabulary rows so the
    # least-squares inverse is not solved on the ontology's own alphabet alone. keep >= d is
    # required by `vocab_side` -- fewer columns than rows leaves the head rank-deficient.
    onto = sorted({t for it in items for t in it["profile"]})
    extra = [int(v) for v in rng.choice(W.shape[0], size=a.pad, replace=False)
             if int(v) not in set(onto)]
    keep = np.array(sorted(set(onto) | set(extra)), dtype=int)
    if len(keep) < d + 8:
        raise ValueError(f"crossing surface {len(keep)} < d+8 = {d + 8}: raise --pad")
    pos = {int(t): i for i, t in enumerate(keep)}
    print(f"crossing surface {len(keep):,} tokens ({len(onto):,} from the ontology, d={d})",
          flush=True)
    side = je.vocab_side(np.eye(d), W, keep, invertible=True)

    def direction(profile: dict) -> np.ndarray:
        v = np.zeros(len(keep), dtype=np.float64)
        for t, w in profile.items():
            v[pos[int(t)]] = w
        v = v - v.mean()
        z = np.asarray(side["inverse"](v[None, :]))[0]
        return z / max(float(np.linalg.norm(z)), 1e-300)

    for it in items:
        it["u"] = direction(it["profile"])
    perm = rng.permutation(len(keep))
    for it in items:
        v = np.zeros(len(keep))
        for t, w in it["profile"].items():
            v[pos[int(t)]] = w
        it["u_shuffled"] = direction({int(keep[perm[pos[int(t)]]]): w
                                      for t, w in it["profile"].items()})

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    state = {"u": None, "amp": 0.0, "at": None}

    def hook(_m, _i, out):
        """Add the direction wherever `state["mask"]` says, at each row's derived local scale.

        WHERE matters more than how much, and the reason is the workspace reading of the
        residual stream. A direction added at the FINAL position of a causal model is attended
        to by nothing -- it is the end of the sequence, so it travels forward only through the
        remaining layers at that one position and is read out immediately. That is a nudge on the
        output, not content entering the shared state. A direction added at a PREFIX position is
        a key and a value for every token after it, which is what "available to the computation"
        means here. The two writes that carried content (2026-08-22, composition and the crossing)
        both used a prefix; the first version of this file used the final position and measured
        nothing.

        Rows are right-padded, so pad columns are excluded from the mask: perturbing a pad would
        change a token the answer never attends to and the arm would silently read as its own
        baseline. Right padding also keeps the default `position_ids` (arange) correct, which
        left padding does not -- the same class of bug as the KV-eviction position error.
        """
        if state["u"] is None:
            return out
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        m = state["mask"]                                    # (B, T) bool
        # amplitude from the stream's own norm over the positions being changed: derived, not set
        n = (h.norm(dim=-1) * m).sum(1) / m.sum(1).clamp(min=1)
        add = state["u"] * (state["amp"] * n)[:, None]       # (B, d)
        return (((h + add[:, None, :] * m[:, :, None].to(h.dtype)),) + tuple(out[1:])) if tup \
            else h + add[:, None, :] * m[:, :, None].to(h.dtype)

    n_prefix = len(tok(prefix, add_special_tokens=False).input_ids)

    def logits_of(its, us=None, layer=0, amp=0.0, text=False, where="prefix"):
        """Final-position logits for a batch, one row per item."""
        asks = [prefix + (ASK_TEXT.format(g=it["gloss"], a=", ".join(it["words"]))
                          if text else ASK.format(g=it["gloss"])) for it in its]
        enc = tok(asks, return_tensors="pt", padding=True, padding_side="right")
        last = enc.attention_mask.sum(1) - 1
        # The shared few-shot prefix must tokenise identically inside the full string, or every
        # index below points at the wrong token.
        head_ids = tok(prefix, add_special_tokens=False).input_ids
        if list(enc.input_ids[0][:n_prefix]) != list(head_ids):
            raise ValueError("the few-shot prefix does not tokenise identically inside the "
                             "prompt: position indices would be wrong")
        cols = torch.arange(enc.input_ids.shape[1])[None, :]
        if where == "final":
            mask = cols == last[:, None]
        elif where == "prefix":                              # first token of the item's own text
            mask = cols == n_prefix
        elif where == "span":                                # the whole question, shots excluded
            mask = (cols >= n_prefix) & (cols <= last[:, None])
        else:
            raise ValueError(f"unknown injection site {where!r}")
        state["mask"] = mask
        state["amp"] = amp
        state["u"] = None if us is None else torch.tensor(np.stack(us), dtype=torch.float32)
        hd = blocks[layer].register_forward_hook(hook) if us is not None else None
        try:
            out = M(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            lg = out.logits[torch.arange(len(its)), last].float().numpy()
        finally:
            if hd is not None:
                hd.remove()
            state["u"] = None
        return lg

    def read(lg, its):
        rows = []
        for k, it in enumerate(its):
            rank = int((lg[k] > lg[k][it["ans_id"]]).sum()) + 1
            cand = it["choices"]
            rows.append((rank, int(cand[int(np.argmax(lg[k][cand]))]) == int(it["ans_id"])))
        return rows

    def probe_all(its, us=None, layer=0, amp=0.0, text=False, where="prefix"):
        rows = []
        for k in range(0, len(its), a.batch):
            chunk = its[k: k + a.batch]
            u = None if us is None else us[k: k + a.batch]
            rows.extend(read(logits_of(chunk, u, layer, amp, text, where), chunk))
        return rows

    # A batching bug would read as "the arm did nothing", which is a result this experiment can
    # plausibly return -- so the batched path is checked against the single-row path before any
    # arm is measured, injected as well as clean.
    for tag, kw in (("clean", {}), ("injected", {"layer": 8, "amp": 0.5})):
        chk = dev[: min(4, len(dev))]
        us = None if not kw else [it["u"] for it in chk]
        many = logits_of(chk, us, **kw)
        for k, it in enumerate(chk):
            one = logits_of([it], None if us is None else [us[k]], **kw)[0]
            gap = float(np.abs(many[k] - one).max())
            if gap > 1e-3:
                raise ValueError(f"batched and single-row logits disagree by {gap:.2e} on the "
                                 f"{tag} path: the batch is not measuring what it reports")
        print(f"  batching verified against single-row logits ({tag})", flush=True)

    def score(rows):
        r = np.array([x[0] for x in rows], dtype=np.float64)
        c = np.array([x[1] for x in rows], dtype=np.float64)
        return {"rank1": float((r == 1).mean()), "rank10": float((r <= 10).mean()),
                "mrr": float((1.0 / r).mean()), "median": float(np.median(r)),
                "choice": float(c.mean())}

    dev_base = score(probe_all(dev))
    print(f"\ndevelopment baseline: {a.choices}-way choice {dev_base['choice']:.1%}  "
          f"rank-1 {dev_base['rank1']:.1%}  rank-10 {dev_base['rank10']:.1%}  "
          f"MRR {dev_base['mrr']:.4f}  median rank {dev_base['median']:.0f}", flush=True)
    if dev_base["rank10"] == 0.0 or dev_base["choice"] <= 1.5 / a.choices:
        raise ValueError("the baseline is at floor: this benchmark cannot discriminate, so no "
                         "arm measured against it would mean anything")
    if dev_base["choice"] >= 0.98:
        raise ValueError(f"the baseline already scores {dev_base['choice']:.1%}: there is no "
                         f"headroom for an arm to show benefit in")

    # Selected on MRR, not on the forced choice. The 20-way choice runs at 90% here, so it has
    # ten points of headroom and one item moves it by 2.5 -- a metric that close to its ceiling
    # cannot order twelve configurations, and the first run of this file proved it: choice picked
    # layer 13 / amp 0.50 on a single extra item while that config's MRR sat 0.035 BELOW baseline.
    # MRR is continuous, unbounded above by the sample, and reads every position of the ranking.
    layers = [int(v) for v in a.layers.split(",")]
    amps = [float(v) for v in a.amps.split(",")]
    sites = [v.strip() for v in a.where.split(",") if v.strip()]
    print(f"\n{'site':>8}{'layer':>6}{'amp':>7}{'MRR':>9}{'vs base':>10}{'rank1':>9}"
          f"{'choice':>9}  (development)")
    grid, best = [], None
    for site in sites:
        for L in layers:
            for amp in amps:
                s = score(probe_all(dev, [it["u"] for it in dev], L, amp, where=site))
                grid.append({"where": site, "layer": L, "amp": amp, **s})
                print(f"{site:>8}{L:>6}{amp:>7.2f}{s['mrr']:>9.4f}"
                      f"{s['mrr'] - dev_base['mrr']:>+10.4f}{s['rank1']:>9.1%}"
                      f"{s['choice']:>9.1%}", flush=True)
                if best is None or s["mrr"] > best["mrr"]:
                    best = grid[-1]
    site, L, amp = best["where"], best["layer"], best["amp"]
    print(f"\nchosen on development: {site} injection, layer {L}, amplitude {amp:.2f} "
          f"(MRR {best['mrr']:.4f} against a {dev_base['mrr']:.4f} baseline, "
          f"{best['mrr'] - dev_base['mrr']:+.4f})", flush=True)
    if best["mrr"] <= dev_base["mrr"]:
        print("  NOTE  no configuration beats the baseline on development. The held-out table "
              "below is reported anyway, and it is the best case for the method, not a search "
              "for one.", flush=True)

    arms = {
        "base": score(probe_all(test)),
        "inject": score(probe_all(test, [it["u"] for it in test], L, amp, where=site)),
        "shuffled": score(probe_all(test, [it["u_shuffled"] for it in test], L, amp, where=site)),
        "wrong": score(probe_all(test, [test[(i + 1) % len(test)]["u"]
                                        for i in range(len(test))], L, amp, where=site)),
        "prompt_text": score(probe_all(test, text=True))}

    print(f"\nHELD OUT, n={len(test)}, {site} injection, layer {L}, amplitude {amp:.2f}")
    print(f"{'arm':>13}{'choice':>9}{'vs base':>10}{'MRR':>9}{'rank1':>9}{'rank10':>9}")
    for k in ("base", "inject", "shuffled", "wrong", "prompt_text"):
        s = arms[k]
        print(f"{k:>13}{s['choice']:>9.1%}{s['choice'] - arms['base']['choice']:>+10.1%}"
              f"{s['mrr']:>9.4f}{s['rank1']:>9.1%}{s['rank10']:>9.1%}", flush=True)

    i, b = arms["inject"]["choice"], arms["base"]["choice"]
    t, sh, wr = (arms["prompt_text"]["choice"], arms["shuffled"]["choice"],
                 arms["wrong"]["choice"])
    # A difference of one item is 1/n. Nothing smaller than that is a difference at all, and
    # nothing under the binomial standard error on n items is worth a verdict.
    se = math.sqrt(max(b * (1.0 - b), 1e-12) / len(test))
    print(f"\none item is {1.0 / len(test):.1%}; the binomial standard error at the baseline "
          f"rate is {se:.1%}", flush=True)
    print()
    if i - b <= se:
        print(f"VERDICT  the entry does not help: {i:.1%} injected against {b:.1%} baseline, a "
              f"{i - b:+.1%} difference inside the {se:.1%} standard error. Selectivity without "
              f"benefit -- the direction moves the model, and not toward a better answer.")
    elif i <= max(sh, wr) + se:
        print(f"VERDICT  the GAIN IS NOT THE CONTENT: {i:.1%} injected beats the {b:.1%} baseline, "
              f"and so do the controls (shuffled {sh:.1%}, wrong entry {wr:.1%}). Perturbing the "
              f"stream at this layer helps regardless of what is injected.")
    elif i <= t + se:
        print(f"VERDICT  the entry helps ({i:.1%} against {b:.1%}, controls {sh:.1%}/{wr:.1%}) and "
              f"pasting the same ancestors into the prompt does as well ({t:.1%}). Injection's "
              f"case is then cost and composition, not accuracy: n entries of prompt text cost n "
              f"times the context, where n directions compose at fixed energy.")
    else:
        print(f"VERDICT  the entry helps AND beats its own text: {i:.1%} injected against a "
              f"{b:.1%} baseline, {t:.1%} for the same ancestors as prompt words, controls "
              f"{sh:.1%} (shuffled) and {wr:.1%} (wrong entry).")

    je.dump(a.out, {"model": a.model, "dev": len(dev), "test": len(test),
                    "surface": int(len(keep)), "ontology_tokens": len(onto),
                    "dev_baseline": dev_base, "grid": grid, "chosen": best,
                    "arms": arms}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError, ImportError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
