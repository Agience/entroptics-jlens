"""The README is the PyPI project page, so its links have to be absolute — and still correct.

PyPI renders `README.md` as the long description and resolves nothing relative, so a
`[the paper](PAPER.md)` link is dead there while looking fine on GitHub. All 25 were rewritten to
absolute `github.com/Agience/entroptics-jlens/blob/main/...` URLs.

That trade only pays if the paths are right. A relative link that is wrong breaks visibly in the
repository; an absolute one that is wrong breaks quietly for a reader who is not in the repository
at all. So this checks every URL against the working tree, which is the thing the URL is a claim
about.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BASE = "https://github.com/Agience/entroptics-jlens"


def links() -> list[tuple[str, str]]:
    """Every markdown link in the README as (text, target)."""
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", README.read_text(encoding="utf-8"))


def repo_links() -> list[str]:
    """The targets that point into this repository, as paths relative to the root."""
    out = []
    for _text, target in links():
        m = re.match(rf"{re.escape(BASE)}/(?:blob|tree)/main/([^)#]+)", target)
        if m:
            out.append(m.group(1))
    return out


def test_the_readme_has_no_relative_links():
    """The reason this file exists. PyPI resolves none of them."""
    bad = [t for _x, t in links()
           if not t.startswith(("http", "mailto:", "#"))]
    assert not bad, f"relative links are dead on PyPI: {bad}"


def test_links_were_found_at_all():
    """The premise. An empty list would make the check below vacuous."""
    found = repo_links()
    assert len(found) >= 20, f"only {len(found)} repository links parsed; the pattern has drifted"


@pytest.mark.parametrize("target", sorted(set(repo_links())))
def test_every_repository_link_points_at_something_that_exists(target: str):
    # The README ships in the distribution; the files it links to are repository files and do
    # not. Checking a link against a tree that was never shipped measures the packaging, not
    # the link.
    if not (ROOT / "research").exists():                       # pragma: no cover - sdist
        pytest.skip("repository tree is not shipped; links are checked from a checkout")
    assert (ROOT / target).exists(), (
        f"README links to {target}, which is not in the tree. An absolute URL that is wrong "
        f"breaks quietly for a reader outside the repository.")


def test_in_page_anchors_match_a_heading():
    """`[the numbers are below](#what-this-does-not-do)` has to land somewhere. GitHub's slug rule
    is lowercase, spaces to hyphens, punctuation dropped."""
    text = README.read_text(encoding="utf-8")
    slugs = set()
    for line in text.splitlines():
        m = re.match(r"^#{1,6} (.+)$", line)
        if m:
            slug = re.sub(r"[^\w\s-]", "", m.group(1).lower()).strip()
            slugs.add(re.sub(r"[\s]+", "-", slug))
    anchors = {t[1:] for _x, t in links() if t.startswith("#")}
    missing = sorted(a for a in anchors if a not in slugs)
    assert not missing, f"anchors with no matching heading: {missing}"


def test_the_base_url_matches_the_declared_repository():
    """If the project moves, these links move with it or they rot."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'Repository\s*=\s*"([^"]+)"', pyproject)
    assert m, "pyproject.toml declares no Repository URL"
    assert m.group(1).rstrip("/") == BASE, (
        f"README links point at {BASE} but pyproject declares {m.group(1)}")
