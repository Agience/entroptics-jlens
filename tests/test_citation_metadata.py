"""`CITATION.cff` and `.zenodo.json` describe the same release as `pyproject.toml`.

Three files state the version, the licence and the repository, and a deposit is made from the two
that nothing else reads. A release cut with a stale `CITATION.cff` mints a DOI whose metadata
disagrees with the package it points at, and a DOI cannot be edited back out of the record.

The CFF is parsed with regexes rather than a YAML library on purpose: PyYAML is not a dependency
of this package, and adding one so that this test can run would put a library in `[dev]` for the
sake of checking a static file.

This package has **no DOI of its own** until it is deposited. The only DOI here is Entroptics',
cited as a reference, and the check below is that the file does not claim one it does not have.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFF = ROOT / "CITATION.cff"
ZENODO = ROOT / ".zenodo.json"
PYPROJECT = ROOT / "pyproject.toml"
# Citation metadata is read from the repository and from the GitHub release, not from the
# distribution, so setuptools does not ship either file. A shipped test that fails for an absent
# repository file reads as a broken package rather than a file that was never distributed.
if not CFF.exists():                                             # pragma: no cover - sdist
    pytest.skip("citation metadata is a repository file and is not shipped",
                allow_module_level=True)

#: The Entroptics deposit. This package consumes it as an engine, so it is cited, not claimed.
ENTROPTICS_DOI = "10.5281/zenodo.21273400"


def cff() -> str:
    return CFF.read_text(encoding="utf-8")


def zenodo() -> dict:
    return json.loads(ZENODO.read_text(encoding="utf-8"))


def field(name: str) -> str:
    """A top-level scalar from the CFF: unindented `key: value`, quotes stripped."""
    m = re.search(rf"^{re.escape(name)}:\s*(.+)$", cff(), re.MULTILINE)
    assert m, f"CITATION.cff has no top-level {name!r}"
    return m.group(1).strip().strip('"')


def pyproject(name: str) -> str:
    m = re.search(rf'^{re.escape(name)} = "([^"]+)"',
                  PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
    assert m, f"pyproject.toml has no {name!r}"
    return m.group(1)


def test_both_files_exist_and_parse():
    """The premise. A missing file would make every check below vacuous."""
    assert CFF.exists(), "CITATION.cff is gone; GitHub's 'Cite this repository' needs it"
    assert ZENODO.exists(), "the Zenodo deposit metadata is gone"
    assert field("cff-version") == "1.2.0"
    assert zenodo()["upload_type"] == "software"


@pytest.mark.parametrize("key", ["version", "license"])
def test_the_citation_files_agree_with_pyproject(key: str):
    want = pyproject(key)
    assert field(key) == want, f"CITATION.cff {key} is {field(key)!r}, pyproject says {want!r}"
    got = zenodo()[key]
    assert got == want, f".zenodo.json {key} is {got!r}, pyproject says {want!r}"


def test_the_repository_is_the_one_pyproject_declares():
    want = pyproject("Repository").rstrip("/")
    assert field("repository-code").rstrip("/") == want
    urls = [r["identifier"].rstrip("/") for r in zenodo()["related_identifiers"]
            if r["scheme"] == "url"]
    assert want in urls, f".zenodo.json points at {urls}, pyproject declares {want}"


def test_the_title_is_the_papers_title():
    """The deposit and the paper are the same work, so they carry the same title. Compared with
    the multiplication sign folded to `x`, because the CFF and the JSON are plain ASCII."""
    paper = (ROOT / "research" / "PAPER.md")
    if not paper.exists():                                       # pragma: no cover - sdist
        pytest.skip("the paper is a repository file and is not shipped")
    want = paper.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    norm = lambda s: s.replace("×", "x").replace("—", "-").strip()   # noqa: E731
    assert norm(field("title")) == norm(want), (
        f"CITATION.cff title:\n  {field('title')}\npaper title:\n  {want}")
    assert norm(zenodo()["title"]) == norm(want), (
        f".zenodo.json title:\n  {zenodo()['title']}\npaper title:\n  {want}")


def test_entroptics_is_cited_as_the_engine():
    """Every resolved-rank count in this work is one of the library's, so the deposit records the
    dependency both ways: a `references` entry in the CFF and an `isDerivedFrom` on the DOI."""
    text = cff()
    assert ENTROPTICS_DOI in text, "CITATION.cff no longer cites the Entroptics DOI"
    assert re.search(r"^references:", text, re.MULTILINE), "the references block is gone"
    rel = {(r["relation"], r["identifier"]) for r in zenodo()["related_identifiers"]}
    assert ("isDerivedFrom", ENTROPTICS_DOI) in rel, (
        f".zenodo.json does not record the Entroptics deposit as its origin: {sorted(rel)}")


#: The Zenodo concept DOI for this work, minted on the v0.1.0 deposit. It resolves to the newest
#: version; the 0.1.0 deposit itself is 10.5281/zenodo.22293930.
CONCEPT_DOI = "10.5281/zenodo.22293929"


def test_the_citation_names_the_deposited_doi():
    """A DOI is minted by depositing and cannot be edited out of the record, so the one named here
    has to be the one that exists. The concept DOI is the citable form: it follows the newest
    version, where a version DOI pins one deposit forever."""
    top = re.search(r"^doi:\s*(.+)$", cff(), re.MULTILINE)
    assert top, "CITATION.cff names no DOI; the work is deposited and should cite it"
    got = top.group(1).strip().strip('"')
    assert got == CONCEPT_DOI, (
        f"CITATION.cff names {got}, not the concept DOI {CONCEPT_DOI}. A version DOI pins one "
        f"deposit, so a citation carrying it stops following the work.")
    assert got != ENTROPTICS_DOI, "that is the engine's DOI, not this work's"


def test_zenodo_json_does_not_hardcode_a_doi():
    """Zenodo assigns a fresh version DOI on every deposit, so a DOI written into the deposit
    metadata would claim an identifier the next release does not have."""
    assert "doi" not in zenodo(), (
        "Zenodo assigns the DOI on deposit; setting one here claims an identifier this repository "
        "did not mint")
