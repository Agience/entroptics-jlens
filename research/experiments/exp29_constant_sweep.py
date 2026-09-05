"""Every numeric constant in the path, found and classified.

A screen is a transfer, and a transfer is only as trustworthy as the arithmetic between the two
sides. A chosen constant anywhere in that path silently sets what crosses: `np.linalg.pinv`'s
default `rcond` turned a round trip that must contract into one with top singular value 1.31, and
it sat on the crossing itself.

So this sweeps the source for numeric literals in load-bearing positions and sorts them:

    DERIVED     computed from the object -- a noise floor from the spectrum, `max(shape) * eps`
                from the dtype. Nothing to yank; the value moves with the data.
    GUARD       a division or log floor, present so an exact zero does not produce a NaN. It sets
                no decision and any value far below the data's scale gives the same answer.
    REPORTED    a default the caller sets and the result records -- `far`, a rank, a seed. It is a
                parameter, not a hidden constant, provided the output carries it.
    CHOSEN      a number that decides something and came from nowhere. These are the ones to
                remove, replace with a derived quantity, or promote to a recorded parameter.

Run it as a standing check. A new CHOSEN constant is a regression.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Only real NUMBER tokens. Two earlier versions of this sweep were wrong in opposite ways: the
# first matched comparison operands and found 2 literals in a package with dozens; the second
# matched every digit in the file and reported measured values quoted in docstrings as unexplained
# constants. `tokenize` sees what the interpreter sees -- no strings, no comments, no prose.
TRIVIAL = {"0", "1", "2", "0.0", "1.0", "2.0", "0.5", "1e-30", "1e-300", "3", "4", "10", "100"}

DERIVED = re.compile(r"noise_floor|finfo|eps|macheps|max\(A\.shape\)|max\(shape\)|"
                     r"transport_spectrum|sv\[0\]|spectrum|singular|shape")
GUARD = re.compile(r"maximum\(|np\.where\(|clip\(|else\s+1\.0|abs\(|denom|"
                   r"1e-1[2-9]|1e-[23][0-9]")
REPORTED = re.compile(r"far|seed|draws|folds|iters|default=|:\s*float\s*=|:\s*int\s*=|"
                      r"add_argument")


def classify(line: str) -> str:
    if DERIVED.search(line):
        return "DERIVED"
    if GUARD.search(line):
        return "GUARD"
    if REPORTED.search(line):
        return "REPORTED"
    return "CHOSEN"


def scan(paths):
    """Numeric literals as the interpreter sees them, with the source line for context."""
    import tokenize
    out = []
    for f in paths:
        src = f.read_text(encoding="utf-8").splitlines()
        with open(f, "rb") as fh:
            for t in tokenize.tokenize(fh.readline):
                if t.type != tokenize.NUMBER or t.string in TRIVIAL:
                    continue
                line = src[t.start[0] - 1] if t.start[0] <= len(src) else ""
                out.append((f, t.start[0], t.string, classify(line), line.strip()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", nargs="*", default=["src/entroptics_jlens"],
                    help="directories to sweep")
    ap.add_argument("--show", default="CHOSEN",
                    help="classes to print in full: comma-separated, or 'all'")
    a = ap.parse_args(argv)

    files = sorted(p for r in a.roots for p in Path(r).rglob("*.py"))
    rows = scan(files)
    counts = {}
    for _, _, _, kind, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    print(f"swept {len(files)} files, {len(rows)} numeric literals in load-bearing positions")
    for kind in ("DERIVED", "GUARD", "REPORTED", "CHOSEN"):
        print(f"  {kind:<10}{counts.get(kind, 0):>4}")

    show = {k.strip().upper() for k in a.show.split(",")} if a.show != "all" else \
        {"DERIVED", "GUARD", "REPORTED", "CHOSEN"}
    print()
    for f, i, lit, kind, code in rows:
        if kind in show:
            print(f"{kind:<9}{f.name}:{i:<5}{lit:<10}{code[:88]}")

    chosen = counts.get("CHOSEN", 0)
    print()
    print(f"{chosen} constants decide something and came from nowhere."
          if chosen else "no unexplained constants in the swept path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
