"""Download a published Jacobian lens.

Kept as the path the experiment scripts and older notes name. It is a shim: the catalogue lives
in ``entroptics_jlens.catalog`` and there is exactly one copy of it, because a second table of
lens paths is a second place for a revision to be wrong. The command line is the same thing:

    entroptics-jlens catalog
    entroptics-jlens fetch pythia-70m gpt2 qwen3.5-4b

    python experiments/fetch_lens.py --list
    python experiments/fetch_lens.py pythia-70m gpt2 qwen3.5-4b

Files land in ``lenses/`` (gitignored). Nothing here reads a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from entroptics_jlens.cli import main as _cli                          # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--list" in argv:
        return _cli(["catalog"])
    keys = [a for a in argv if not a.startswith("-")]
    rest = []
    if "--dir" in argv:
        rest = ["--dir", argv[argv.index("--dir") + 1]]
    return _cli(["fetch", *keys, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
