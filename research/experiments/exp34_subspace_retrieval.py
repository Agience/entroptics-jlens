"""Retrieval by subspace overlap rather than by a cosine between two points.

Every efficiency read in this repo has been about model internals, and all of them failed. The
coverage measure is not about model internals: it is a statement about two frames, and it does not
care that they came from a transformer.

Standard dense retrieval discards almost everything. A document of 200 tokens becomes one
mean-pooled vector, and relevance is a cosine between two points in R^d. But a passage is not a
point: its token frame occupies a SUBSPACE, and two texts about the same thing should span
overlapping subspaces even where their centroids sit apart. `coverage` measures exactly that, and
unlike a cosine it carries an analytic null -- `k_t / d` is what the overlap would be by chance for
a readout of that size, so a score can be compared against what it would be for unrelated text.

    baseline   cosine between mean-pooled frames -- what dense retrieval does
    subspace   coverage of the query's resolved subspace by the document's, against its null
    truth      BeIR/scifact relevance judgments: 339 labelled pairs over 5183 documents,
               a standard benchmark this work played no part in constructing

Evaluated as re-ranking, which is how a second-stage scorer is actually deployed: take the top `n`
by cosine, re-order by each method, and score with nDCG@10 and recall@10 against the judgments. A
re-ranker that cannot beat the ordering it was handed is worth nothing, so the baseline is the
first stage itself.

No threshold anywhere: the resolved rank of each frame comes from its own noise floor, and the
null comes from the dimensions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402


def ndcg_at_k(ranked_ids, relevant, k=10):
    """nDCG@k with binary relevance, the standard BeIR reporting metric."""
    gains = [1.0 / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant]
    ideal = [1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))]
    return float(sum(gains) / sum(ideal)) if ideal else 0.0


def recall_at_k(ranked_ids, relevant, k=10):
    return float(len(set(ranked_ids[:k]) & relevant) / len(relevant)) if relevant else 0.0


def mrr(ranked_ids, relevant):
    """Reciprocal rank of the first hit -- sensitive to the very top, where nDCG@10 is not.

    Measured: over a top-50 candidate set whose recall@10 is already 0.85, "pick the longest
    document" scores nDCG@10 0.7284 against a real ranker's 0.7342. A metric where the dumbest
    ordering lands within 0.006 of the best cannot separate methods, and a null measured through
    it is a statement about the metric.
    """
    for i, d in enumerate(ranked_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--queries", type=int, default=60, help="labelled queries to evaluate")
    ap.add_argument("--pool", type=int, default=1200, help="corpus documents in the pool")
    ap.add_argument("--rerank", type=int, default=50, help="first-stage depth to re-order")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/subspace_retrieval.json"))
    a = ap.parse_args(argv)

    import torch
    import transformers
    from datasets import load_dataset
    torch.set_grad_enabled(False)

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    rel = {}
    for r in qrels:
        if int(r["score"]) > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    qtext = {str(q["_id"]): q["text"] for q in queries}
    have = [q for q in rel if q in qtext][: a.queries]

    # the pool must contain every judged document, or the ceiling is capped by sampling
    needed = {d for q in have for d in rel[q]}
    ctext, cids = {}, []
    for row in corpus:
        cid = str(row["_id"])
        ctext[cid] = (row["title"] + ". " + row["text"]).strip()
        cids.append(cid)
    pool = list(needed) + [c for c in cids if c not in needed][: max(a.pool - len(needed), 0)]
    print(f"{len(have)} queries, {len(pool)} documents in pool "
          f"({len(needed)} judged relevant), re-ranking depth {a.rerank}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(a.encoder)
    enc = transformers.AutoModel.from_pretrained(a.encoder).eval().float()

    def frame(text):
        """The token-level activation frame -- the object a mean-pool throws away."""
        ids = tok(text, return_tensors="pt", truncation=True, max_length=a.tokens)
        out = enc(**ids).last_hidden_state[0]
        m = ids["attention_mask"][0].bool()
        return out[m].numpy().astype(np.float64)

    print("encoding pool", flush=True)
    frames = {}
    for i, cid in enumerate(pool):
        frames[cid] = frame(ctext[cid])
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(pool)}", flush=True)
    means = {c: f.mean(0) for c, f in frames.items()}
    M = np.stack([means[c] for c in pool])
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-30)

    rows = []
    for qi, q in enumerate(have):
        qf = frame(qtext[q])
        qm = qf.mean(0)
        qm = qm / max(np.linalg.norm(qm), 1e-30)
        cos = Mn @ qm
        order = np.argsort(-cos)[: a.rerank]
        cand = [pool[i] for i in order]

        # Subspace overlap, at MATCHED entropy density.
        #
        # A query frame is ~15 tokens x 384 dims and a document frame ~128 x 384. Their resolved
        # ranks are set by their own aspect ratios, so the query's floor is dominated by having
        # almost no samples per dimension while the document's is not. Comparing a subspace
        # resolved at T/d ~ 0.04 against one resolved at T/d ~ 0.33 compares two objects whose
        # thresholds mean different things -- the same error as estimating structure in 768
        # dimensions from 128 rows.
        #
        # Matching is the fix: both sides are read at the SHORTER frame's token count, so the two
        # floors are computed at one density and the overlap is between comparable subspaces.
        cov, dlen = [], []
        rs = np.random.default_rng(qi)
        for c in cand:
            df = frames[c]
            n = min(len(qf), len(df))
            qs = qf if len(qf) == n else qf[np.sort(rs.choice(len(qf), n, replace=False))]
            dsub = df if len(df) == n else df[np.sort(rs.choice(len(df), n, replace=False))]
            r = je.coverage(qs, dsub, far=a.far)
            cov.append(r.excess)
            dlen.append(len(df))
        resub = [c for _, c in sorted(zip(cov, cand), key=lambda t: -t[0])]

        # Length-residualised. On TruthfulQA a coverage score correlated -0.303 with candidate
        # length while log-probability correlated +0.355, and a "pick the longest" baseline beat
        # both -- the read was sorting by a nuisance variable. Document length has little to do
        # with relevance, so the same dependence here would drown whatever signal exists rather
        # than flatter it. Regressing coverage on length within the candidate set removes it.
        cv, lv = np.asarray(cov, float), np.asarray(dlen, float)
        if lv.std() > 1e-9 and len(cv) >= 3:
            res = cv - np.polyval(np.polyfit(lv, cv, 1), lv)
        else:
            res = cv - cv.mean()
        rescr = [c for _, c in sorted(zip(res, cand), key=lambda t: -t[0])]
        corr = float(np.corrcoef(cv, lv)[0, 1]) if cv.std() > 0 and lv.std() > 0 else 0.0

        R = rel[q]
        # the dumbest baseline, run before believing a subtle one
        bylen = [c for _, c in sorted(zip(dlen, cand), key=lambda t: -t[0])]
        rows.append({
            "query": q, "n_rel": len(R), "cov_len_corr": corr,
            "cos_ndcg": ndcg_at_k(cand, R), "sub_ndcg": ndcg_at_k(resub, R),
            "cos_mrr": mrr(cand, R), "sub_mrr": mrr(resub, R),
            "res_mrr": mrr(rescr, R), "len_mrr": mrr(bylen, R),
            "cos_ndcg1": ndcg_at_k(cand, R, 1), "sub_ndcg1": ndcg_at_k(resub, R, 1),
            "res_ndcg1": ndcg_at_k(rescr, R, 1),
            "res_ndcg": ndcg_at_k(rescr, R), "len_ndcg": ndcg_at_k(bylen, R),
            "cos_recall": recall_at_k(cand, R), "sub_recall": recall_at_k(resub, R),
            "res_recall": recall_at_k(rescr, R),
        })
        if (qi + 1) % 10 == 0:
            print(f"  query {qi + 1}/{len(have)}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows])

    print()
    print(f"{'method':>22}{'nDCG@10':>10}{'nDCG@1':>9}{'MRR':>8}{'recall@10':>12}")
    print(f"{'cosine (first stage)':>22}{col('cos_ndcg').mean():>10.4f}"
          f"{col('cos_ndcg1').mean():>9.4f}{col('cos_mrr').mean():>8.4f}"
          f"{col('cos_recall').mean():>12.4f}")
    print(f"{'subspace coverage':>22}{col('sub_ndcg').mean():>10.4f}"
          f"{col('sub_ndcg1').mean():>9.4f}{col('sub_mrr').mean():>8.4f}"
          f"{col('sub_recall').mean():>12.4f}")
    print(f"{'coverage, length-free':>22}{col('res_ndcg').mean():>10.4f}"
          f"{col('res_ndcg1').mean():>9.4f}{col('res_mrr').mean():>8.4f}"
          f"{col('res_recall').mean():>12.4f}")
    print(f"{'pick longest (control)':>22}{col('len_ndcg').mean():>10.4f}"
          f"{'':>9}{col('len_mrr').mean():>8.4f}{'':>12}")
    print()
    print(f"  corr(coverage, doc length) within candidates: "
          f"{col('cov_len_corr').mean():+.3f}")

    rng = np.random.default_rng(0)
    d = col("res_mrr") - col("cos_mrr")
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print()
    print(f"length-free coverage vs cosine, MRR {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")
    je.dump(a.out, {"encoder": a.encoder, "queries": len(rows), "pool": len(pool),
                    "rerank_depth": a.rerank,
                    "cos_ndcg": float(col("cos_ndcg").mean()),
                    "sub_ndcg": float(col("sub_ndcg").mean()),
                    "res_ndcg": float(col("res_ndcg").mean()),
                    "len_ndcg": float(col("len_ndcg").mean()),
                    "cov_len_corr": float(col("cov_len_corr").mean()),
                    "cos_recall": float(col("cos_recall").mean()),
                    "sub_recall": float(col("sub_recall").mean()),
                    "delta": {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi)},
                    "rows": rows}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
