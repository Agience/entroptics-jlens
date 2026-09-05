"""The two lexicons are already connected, by a key that is already in the store.

`_archive/2026-08-25-doc-triage/WHERE-WE-STAND.md` names this as the second education priority and as
item 2 on John's desk: Princeton and OEWN both describe English, sit 1.5 apart in the geometry
(three correlation lengths at xi = 0.465), and the served SSE index holds only OEWN -- so a query
whose meaning lands on Princeton cannot reach the documents that are searchable. The remedy on
offer there is a `(pos, lemma set)` join that pairs 95.6% of them.

A lemma-set join is a rule about surface FORM. It is wrong exactly where the two lexicons disagree
about naming, which is the interesting 5%: OEWN deliberately retitled `alderman` to `alderperson`,
`baggageman` to `baggage attendant`, `aircraftsman` to `aircraft officer`, and corrected
`antonius pius` to `antoninus pius`. Those pairs share no lemma at all, so a form join cannot see
them, and they are the cases where a caller most needs the bridge.

There is a better key sitting unused in the same store, and finding it needed one correction first.

    The `wn-` namespace is NOT 555,595 Princeton synsets. It is 117,659 Princeton ENGLISH synsets
    plus 437,936 Open Multilingual WordNet synsets in fifteen non-English languages (fi, ca, es,
    id, zsm, it, eu, hr, gl, el, iwn, sv, bg, is, sq).

Those OMW rows are the bridge. All but 6 carry an `ili` -- an Interlingual Index id, the
identifier WordNets across languages agree on -- and each one is NAMED by its Princeton 3.0
offset (`wn-omw-bg-00001740-v`). Every OEWN synset carries an `ili` too. So:

    princeton synset --nltk--> (offset, pos) --OMW row--> ili --> OEWN synset

No string matching anywhere on the path. The lemma sets are then free to serve as an INDEPENDENT
check on a join they took no part in.

Two facts make the middle step exact rather than approximate, and both are measured here rather
than assumed: no (offset, pos) key in OMW carries more than one ili, and one ili in OEWN names more
than one synset. The one place the naive form of the walk fails is satellite adjectives -- Princeton
`pos` 's' -- because OMW ids carry no 's': PWN keeps satellites in the ADJECTIVE offset space. Left
unhandled that is a 0.0% pairing rate on 10,693 synsets, and adjectives are precisely the tier
WHERE-WE-STAND measures as worst served (`modifier`, 2/6).

Read-only against the live shard. The store is served and written by others; this opens it
`mode=ro` with `query_only`, and takes no lock a writer would wait on.
"""
from __future__ import annotations

import argparse
import os
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

# The lattice store is a multi-GB file on the operator's own disk, so its location is configuration,
# not a constant. This used to default to one machine's `D:` drive, which is a path that resolves
# nowhere else and publishes where its author keeps their data. `EMBER_SQLITE_DIR` is the variable
# the rest of this workspace already uses for the same store; when it is unset there is no default
# and argparse requires `--db`, which fails with a usage message rather than a confusing
# "unable to open database file" against a path the caller never chose.
def _default_db():
    d = os.environ.get("EMBER_SQLITE_DIR")
    return str(Path(d) / "lattice.db") if d else None

OMW_ID = re.compile(r"^wn-omw-([a-z]{2,3})-([0-9]{8})-([nvasr])$")


def lemmas_of(doc: dict) -> frozenset:
    return frozenset(x.lower().replace("_", " ") for x in doc.get("lemmas", ()))


def scan(con, lo: str, hi: str):
    for vid, doc in con.execute(
            "SELECT id, doc FROM vertex WHERE id>=? AND id<?", (lo, hi)):
        yield vid, json.loads(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=_default_db(), required=_default_db() is None,
                    help="lattice store (default: $EMBER_SQLITE_DIR/lattice.db)")
    ap.add_argument("--out", type=Path, default=Path("results/lexicon_bridge.json"))
    ap.add_argument("--write-pairs", type=Path, default=None,
                    help="write the paired ids as JSONL for the genesis side to consume")
    a = ap.parse_args(argv)

    from nltk.corpus import wordnet as wn                              # noqa: E402

    uri = "file:///" + str(a.db).replace("\\", "/").lstrip("/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=300)
    con.execute("PRAGMA query_only=1")

    ili_to_oewn = collections.defaultdict(list)
    oewn_lemmas = {}
    for vid, doc in scan(con, "wn-oewn-", "wn-oewn."):
        ili = doc.get("ili")
        if not ili:
            raise ValueError(f"{vid} has no ili: the bridge's key is not present on OEWN")
        ili_to_oewn[ili].append(vid)
        oewn_lemmas[vid] = lemmas_of(doc)

    off_to_ili = {}
    langs = collections.Counter()
    ambiguous = 0
    for vid, doc in scan(con, "wn-omw-", "wn-omw."):
        m = OMW_ID.match(vid)
        if not m:
            continue
        langs[m.group(1)] += 1
        ili = doc.get("ili")
        if not ili:
            continue
        key = (m.group(2), m.group(3))
        prev = off_to_ili.setdefault(key, ili)
        if prev != ili:
            ambiguous += 1

    dup_ili = sum(1 for v in ili_to_oewn.values() if len(v) > 1)
    print(f"OEWN     {len(oewn_lemmas):,} synsets, {len(ili_to_oewn):,} distinct ili, "
          f"{dup_ili} ili naming more than one synset")
    print(f"OMW      {sum(langs.values()):,} rows in {len(langs)} non-English languages, "
          f"{len(off_to_ili):,} (PWN offset, pos) keys, {ambiguous} ambiguous")
    print(f"         {' '.join(f'{k}:{v:,}' for k, v in langs.most_common())}")
    if ambiguous:
        raise ValueError(f"{ambiguous} OMW keys carry conflicting ili: the walk is not a function")

    princeton = []
    for vid, doc in scan(con, "wn-", "wn."):
        if vid.startswith(("wn-oewn-", "wn-omw-")):
            continue
        princeton.append((vid, doc.get("pos"), lemmas_of(doc)))
    print(f"Princeton {len(princeton):,} English synsets\n")

    stat = collections.Counter()
    by_pos = collections.defaultdict(collections.Counter)
    renamed, pairs = [], []
    for vid, pos, lem in princeton:
        try:
            syn = wn.synset(vid[3:])
        except Exception:
            stat["no-nltk-synset"] += 1
            by_pos[pos]["no-nltk-synset"] += 1
            continue
        off = f"{syn.offset():08d}"
        # PWN keeps satellite adjectives in the adjective offset space and OMW ids carry no 's'.
        ili = off_to_ili.get((off, syn.pos()))
        if ili is None and syn.pos() == "s":
            ili = off_to_ili.get((off, "a"))
        if ili is None:
            stat["no-omw-key"] += 1
            by_pos[pos]["no-omw-key"] += 1
            continue
        hit = ili_to_oewn.get(ili)
        if not hit:
            stat["ili-absent-from-oewn"] += 1
            by_pos[pos]["ili-absent-from-oewn"] += 1
            continue
        other = oewn_lemmas[hit[0]]
        agree = "exact" if other == lem else ("overlap" if other & lem else "disjoint")
        stat[agree] += 1
        by_pos[pos][agree] += 1
        pairs.append({"princeton": vid, "oewn": hit[0], "ili": ili, "lemma_agreement": agree})
        if agree == "disjoint" and len(renamed) < 12:
            renamed.append((vid, sorted(lem), hit[0], sorted(other)))

    n = len(princeton)
    paired = stat["exact"] + stat["overlap"] + stat["disjoint"]
    for k, c in stat.most_common():
        print(f"  {k:<22}{c:>8,}{c / n:>8.2%}")
    print(f"\n  PAIRED                {paired:>8,}{paired / n:>8.2%}   "
          f"lemma-verified {(stat['exact'] + stat['overlap']) / paired:.2%}")

    print(f"\n{'pos':>4}{'synsets':>10}{'paired':>9}{'exact':>9}{'overlap':>9}"
          f"{'disjoint':>10}{'unpaired':>10}")
    pos_rows = {}
    for pos in ("n", "v", "a", "s", "r"):
        c = by_pos[pos]
        tot = sum(c.values())
        if not tot:
            continue
        got = c["exact"] + c["overlap"] + c["disjoint"]
        pos_rows[pos] = {"synsets": tot, "paired": got, "exact": c["exact"],
                         "overlap": c["overlap"], "disjoint": c["disjoint"]}
        print(f"{pos:>4}{tot:>10,}{got / tot:>9.1%}{c['exact'] / tot:>9.1%}"
              f"{c['overlap'] / tot:>9.1%}{c['disjoint'] / tot:>10.1%}{tot - got:>10,}")

    print(f"\nthe {stat['disjoint']} pairs sharing NO lemma -- what a form join cannot see:")
    for p, pl, o, ol in renamed:
        print(f"  {p:<28} {', '.join(pl)[:34]:<34} -> {', '.join(ol)[:34]}")

    print(f"\n{n - paired:,} Princeton synsets ({(n - paired) / n:.2%}) remain unpaired: they have "
          f"no translation in any of the {len(langs)} OMW languages, so no ili reaches them.")

    if a.write_pairs:
        a.write_pairs.parent.mkdir(parents=True, exist_ok=True)
        with a.write_pairs.open("w", encoding="utf-8", newline="\n") as fh:
            for p in pairs:
                fh.write(json.dumps(p) + "\n")
        print(f"wrote {len(pairs):,} pairs to {a.write_pairs}")

    je.dump(a.out, {"db": str(a.db), "princeton_english": n, "oewn": len(oewn_lemmas),
                    "omw_rows": sum(langs.values()), "omw_languages": dict(langs),
                    "omw_keys": len(off_to_ili), "omw_ambiguous": ambiguous,
                    "oewn_duplicate_ili": dup_ili, "paired": paired,
                    "counts": dict(stat), "by_pos": pos_rows,
                    "renamed_examples": [{"princeton": p, "princeton_lemmas": pl,
                                          "oewn": o, "oewn_lemmas": ol}
                                         for p, pl, o, ol in renamed]}, complete=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, sqlite3.Error, je.IncompleteResults, ValueError, ImportError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
