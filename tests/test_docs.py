"""Lint the prose the same way the code is linted.

Every defect checked here has actually occurred in this repo, and each arrived the same way: a
patch applied through a non-raw Python string, where `\t`, `\r` and `\n` inside the replacement
were interpreted as escapes before reaching the file. `$d_{\text{model}}$` became `$d_{<TAB>ext…`,
`$\rho=+0.72$` became a carriage return that split the line, and a docstring terminator was
joined to the line after it.

None of it is visible in a diff at a glance, and all of it survives to the rendered document.
"""
from pathlib import Path

import pytest

#: The backslash, built rather than written: this file is about defects that arrive when a patch
#: goes through a non-raw Python string, and a literal one here is the same hazard.
BS = chr(92)

#: Every markdown file in the tree, found by walking it rather than by naming two directories.
#: A named list only lints the places someone remembered, and the point of this file is to catch
#: what nobody is watching -- `experiments/README.md` was written outside the old two-glob list
#: and was unlinted on arrival.
ROOT = Path(__file__).resolve().parents[1]
PRUNE = {".git", ".venv", "node_modules", "__pycache__", "build", "dist", "lenses", "out",
         "results", "streams", ".pytest_cache", ".ruff_cache"}
DOCS = sorted(p for p in ROOT.rglob("*.md")
              if not (PRUNE & set(p.relative_to(ROOT).parts)))
TAB, CR = chr(9), chr(13)


def _lines(p):
    return p.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_literal_tabs(doc):
    """A tab in prose is `\text` or `\times` that lost its backslash."""
    bad = [i for i, l in enumerate(_lines(doc), 1) if TAB in l]
    assert not bad, f"{doc.name}: literal tab on line(s) {bad} -- an eaten backslash"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_stray_carriage_returns(doc):
    """A bare CR inside the text is `\r` from `\rho` that lost its backslash and split a line."""
    raw = doc.read_bytes().decode("utf-8")
    stray = raw.replace(CR + "\n", "").count(CR)
    assert stray == 0, f"{doc.name}: {stray} stray carriage return(s) -- an eaten backslash"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_inline_math_is_balanced(doc):
    """An odd number of `$` on a line means a broken formula, usually a split one."""
    bad = [(i, l[:60]) for i, l in enumerate(_lines(doc), 1)
           if not l.strip().startswith("$$") and l.count("$") % 2]
    assert not bad, f"{doc.name}: unbalanced inline math at {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_orphaned_line_fragments(doc):
    """A line beginning mid-token is the tail of one a stray escape split in two."""
    frags = ("ho=", "ext{", "imes", "n.join", "approx", "cos(", "lVert")
    bad = [(i, l[:50]) for i, l in enumerate(_lines(doc), 1)
           if l and l[0].islower() and l.startswith(frags)]
    assert not bad, f"{doc.name}: orphaned fragment starting a line at {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_tables_have_consistent_column_counts(doc):
    """A markdown table whose rows disagree on column count renders wrong and usually means a
    row was edited without its header."""
    lines, bad, i = _lines(doc), [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace(" ", "")) <= set("|-:"):
            # `\|` is an escaped pipe inside a cell and does not split it.
            def cols(t):
                return t.replace(BS + "|", "").count("|")
            width = cols(lines[i])
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                if cols(lines[j]) != width:
                    bad.append((j + 1, lines[j][:50]))
                j += 1
            i = j
        else:
            i += 1
    assert not bad, f"{doc.name}: ragged table row(s) at {bad}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_replacement_characters(doc):
    """U+FFFD is a byte sequence that failed to decode -- a file written through the wrong
    encoding. Cheap to check and invisible in a terminal that cannot render the original."""
    s = doc.read_text(encoding="utf-8")
    n = s.count("\ufffd")
    assert n == 0, f"{doc.name}: {n} replacement character(s) -- an encoding round-trip failed"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    """A markdown link to a path in this repo must point at something that exists. Files get
    renamed; a dead link in the front door is worse than no link."""
    import re
    root = doc.parent
    bad = []
    for text, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (root / target.split("#")[0]).resolve()
        if not path.exists():
            bad.append((text, target))
    assert not bad, f"{doc.name}: dead relative link(s) {bad}"
