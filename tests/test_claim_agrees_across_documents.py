"""The claim table exists in two documents. They must not drift apart.

`README.md` §"The claim" and `PAPER.md` §1 both carry the same ten rows. Two copies of a
measurement is two places for it to go stale, and the failure mode is quiet: a reader who follows
the README to the paper finds different numbers under the same heading and cannot tell which is
current.

The tables are formatted differently on purpose -- the paper is LaTeX, the README is plain -- so
this compares the parsed values rather than the text, which is the only comparison that means
anything across two renderings.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAPER = ROOT / "research" / "PAPER.md"
# The sdist ships this suite but not every repository file it reads. A shipped test that fails
# for an absent repository file reads as a broken package rather than a file that was never
# part of the distribution, so the module skips instead.
if not PAPER.exists():                                           # pragma: no cover - sdist
    pytest.skip("the paper is a repository file and is not shipped", allow_module_level=True)

#: model -> (width, identity share, K(J), K(J - alpha I))
Row = tuple[int, float, int, int]


def _rows(path: Path, header: str) -> dict[str, Row]:
    text = path.read_text(encoding="utf-8")
    start = text.index(header)
    block = text[start:text.index("\n\n", start)]
    out: dict[str, Row] = {}
    for line in block.splitlines():
        m = re.match(r"\|\s*([\w.\-]+)\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}"
                     r"\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), float(m.group(3)),
                               int(m.group(4)), int(m.group(5)))
    return out


def readme_rows() -> dict[str, Row]:
    return _rows(README, "| model | width | identity share |")


def paper_rows() -> dict[str, Row]:
    return _rows(PAPER, "| model | $d$ | identity share |")


def test_both_tables_were_parsed():
    """The premise. Two empty parses would compare equal and prove nothing."""
    r, p = readme_rows(), paper_rows()
    assert len(r) == 10, f"README parsed {len(r)} rows: {sorted(r)}"
    assert len(p) == 10, f"PAPER parsed {len(p)} rows: {sorted(p)}"


def test_the_claim_table_is_the_same_in_both_documents():
    r, p = readme_rows(), paper_rows()
    assert set(r) == set(p), f"models differ: README {sorted(set(r) - set(p))}, " \
                             f"PAPER {sorted(set(p) - set(r))}"
    for model in sorted(r):
        assert r[model] == p[model], (
            f"{model}: README says (width, identity, K(J), K(J-aI)) = {r[model]}, "
            f"PAPER says {p[model]}")


def test_the_paper_credits_entroptics_for_the_measurement():
    """The claim is a correction to an Entroptics read, and the paper has to say so. It named the
    library once in 735 lines after an earlier restructure dropped the framing that carried the
    attribution."""
    text = PAPER.read_text(encoding="utf-8")
    assert text.count("ntroptics") >= 10, (
        f"the paper names entroptics {text.count('ntroptics')} times; the measurement it reports "
        f"is the library's and the attribution has gone thin again")
    assert "## 2. What Entroptics supplies" in text, "the section evaluating the library is gone"
    for token in ("noise_floor", "null_providers", "mode_significance"):
        assert token in text, f"the paper no longer names {token}, which produces every K in it"


def test_the_citation_carries_the_doi_and_the_installable_package():
    """A reader has to be able to get the exact thing the numbers came from."""
    text = PAPER.read_text(encoding="utf-8")
    assert "10.5281/zenodo.21273400" in text, "the Zenodo DOI is missing"
    assert "pypi.org/project/entroptics" in text, "the PyPI source is missing"
    assert "0.2.1" in text, "the pinned version the measurements used is missing"


def test_the_package_depends_on_the_published_entroptics():
    """The paper cites PyPI; the package must actually resolve from there rather than a sibling
    checkout, or the citation describes something a reader cannot reproduce."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Scoped to the dependencies array. A bare `"entroptics"` also appears in `keywords`, and the
    # first version of this test matched that and reported the dependency as unpinned.
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", pyproject, re.MULTILINE | re.DOTALL)
    assert deps, "pyproject.toml has no dependencies array"
    m = re.search(r'"entroptics([><=~!][^"]*)"', deps.group(1))
    assert m, f"entroptics is not a declared dependency, or is unpinned: {deps.group(1).strip()}"
