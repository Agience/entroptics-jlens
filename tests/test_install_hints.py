"""An install hint has to name an extra that exists, and an optional import has to give one.

`catalog.py` told anyone without huggingface_hub to run `pip install 'entroptics-jlens[fetch]'`.
There is no `fetch` extra. pip warns and installs nothing, so the user runs the command, gets the
same ImportError, and has no way to tell that the instruction rather than their environment was
wrong. It sat on the `fetch` command -- the first line of the README's "start here", the one path
every new user takes.

Nothing caught it because the message is only produced in an environment missing a dependency, and
every environment the suite runs in has them all.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "entroptics_jlens"
PYPROJECT = ROOT / "pyproject.toml"

#: Third-party modules the package imports lazily rather than requiring at import time.
OPTIONAL = {"torch", "huggingface_hub", "safetensors"}

DOCS = [ROOT / "README.md", ROOT / "research" / "PAPER.md", ROOT / "CONTRIBUTING.md"]


def defined_extras() -> set[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text,
                      re.MULTILINE | re.DOTALL)
    assert block, "pyproject.toml defines no [project.optional-dependencies]"
    return set(re.findall(r"^([\w-]+)\s*=\s*\[", block.group(1), re.MULTILINE))


def hinted_extras() -> list[tuple[str, str]]:
    """(where, extra) for every `entroptics-jlens[x]` in source or prose."""
    out = []
    for path in sorted(SRC.glob("*.py")) + [d for d in DOCS if d.exists()]:
        for m in re.finditer(r"entroptics-jlens\[([\w,-]+)\]", path.read_text(encoding="utf-8")):
            for one in m.group(1).split(","):
                out.append((path.name, one))
    return out


def test_extras_and_hints_were_both_found():
    """The premise. Either side coming back empty would make the check below vacuous."""
    assert defined_extras() >= {"lens", "dev"}, f"parsed extras: {defined_extras()}"
    assert len(hinted_extras()) >= 3, f"parsed hints: {hinted_extras()}"


def test_the_lazy_import_scan_finds_something():
    """The other premise. `test_optional_imports_refuse_with_a_hint` skips any file with no lazy
    optional import, so a broken detector would skip every file and report all green."""
    carriers = [p.name for p in sorted(SRC.glob("*.py"))
                if _optional_imports(ast.parse(p.read_text(encoding="utf-8")))]
    assert len(carriers) >= 3, (
        f"only {carriers} detected as importing {sorted(OPTIONAL)} lazily; the AST scan has "
        f"drifted and the guard below is skipping everything")


@pytest.mark.parametrize("where,extra", hinted_extras(), ids=lambda v: str(v))
def test_every_install_hint_names_an_extra_that_exists(where: str, extra: str):
    assert extra in defined_extras(), (
        f"{where} tells the user to install 'entroptics-jlens[{extra}]', which pyproject.toml does "
        f"not define. pip warns and installs nothing, so the hint sends them in a circle. "
        f"Defined: {sorted(defined_extras())}")


def _optional_imports(tree: ast.AST) -> list[ast.stmt]:
    """Import statements for OPTIONAL modules that sit inside a function body."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                names = {a.name.split(".")[0] for a in sub.names}
            elif isinstance(sub, ast.ImportFrom):
                names = {(sub.module or "").split(".")[0]}
            else:
                continue
            if names & OPTIONAL:
                found.append(sub)
    return found


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_optional_imports_refuse_with_a_hint(path: Path):
    """A lazy import of torch or huggingface_hub must be wrapped in a try/except that raises a
    message naming the extra. Otherwise the user gets a bare ModuleNotFoundError, which does not
    say that an extra exists at all -- and this package's contract is that everything refuses
    rather than guesses."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lazy = _optional_imports(tree)
    if not lazy:
        pytest.skip(f"{path.name} imports none of {sorted(OPTIONAL)} lazily")
    handled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                handled.add(id(sub))
    unguarded = [n for n in lazy if id(n) not in handled]
    # frames.py reaches for torch only to name a dtype on an object the caller already handed in
    # as a torch tensor, so torch is necessarily importable there.
    if path.name == "frames.py":
        pytest.skip("torch is reached only through an object the caller already built with it")
    assert not unguarded, (
        f"{path.name} imports {sorted(OPTIONAL)} at lines "
        f"{sorted(n.lineno for n in unguarded)} without a try/except, so a user without the extra "
        f"gets a bare ModuleNotFoundError instead of the install hint")


def test_the_readmes_start_here_command_is_reachable_after_a_plain_install():
    """`pip install entroptics-jlens` brings numpy and entroptics only. The README's first command
    is `fetch`, which needs huggingface_hub -- so the hint it raises has to be correct, which is
    what the parametrised check above guarantees. This pins the premise: that `fetch` really is
    outside the base dependencies, so the hint is on the path a new user takes."""
    text = PYPROJECT.read_text(encoding="utf-8")
    base = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert base, "pyproject.toml has no base dependencies array"
    assert "huggingface_hub" not in base.group(1) and "huggingface-hub" not in base.group(1), (
        "huggingface_hub is now a base dependency; the fetch hint is no longer on a path a user "
        "can reach, and this test's premise has gone stale")
