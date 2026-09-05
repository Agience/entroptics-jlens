"""Structural guards over the experiment scripts.

`experiments/` is part of the published tree -- the README points a reader at it as the place
every figure is re-derived from -- so a script there is something a stranger will run.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "research" / "experiments").glob("*.py"))
# experiments/ is a repository directory and is not part of the distribution. Without this the
# whole module fails on an unpacked sdist for a directory that was never shipped.
if not EXPERIMENTS:                                              # pragma: no cover - sdist
    pytest.skip("experiments/ is a repository directory and is not shipped",
                allow_module_level=True)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _is_script(tree: ast.Module) -> bool:
    """A file that runs something when executed, as opposed to a helper other scripts import."""
    return any(isinstance(n, ast.If) and ast.unparse(n.test).startswith("__name__")
               for n in tree.body)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_every_runnable_script_parses_its_arguments(path: Path):
    """A script without an ArgumentParser ignores `--help` and runs.

    That is not hypothetical here: it is the recorded cause of a truncated results file --
    a `--help` re-ran an experiment, a timeout killed it at layer 26 of 31, and a later script
    read the well-formed JSON it left behind as a finished sweep (`entroptics_jlens.results`).
    `exp6_coupling_control.py` still had the shape on 2026-09-02, where `--help` would have
    downloaded a model before doing anything else.

    Helper modules are exempt by not having a `__main__` guard, which is the same property that
    makes them not runnable.
    """
    tree = _tree(path)
    if not _is_script(tree):
        pytest.skip(f"{path.name} is imported, not run: no __main__ guard")
    parses = any(isinstance(n, ast.Attribute) and n.attr == "ArgumentParser"
                 for n in ast.walk(tree))
    delegates = any(isinstance(n, ast.ImportFrom) and n.module and "cli" in n.module
                    for n in ast.walk(tree))
    assert parses or delegates, (
        f"{path.name} has a __main__ guard and no ArgumentParser, so `--help` is ignored and the "
        f"experiment runs. Add a parser, or hand the arguments to one that has it.")


def test_the_guard_can_tell_a_script_from_a_helper():
    """The premise. If nothing in experiments/ were classified either way, the test above would
    pass by skipping everything."""
    trees = {p.name: _tree(p) for p in EXPERIMENTS}
    scripts = [n for n, t in trees.items() if _is_script(t)]
    helpers = [n for n, t in trees.items() if not _is_script(t)]
    assert len(scripts) > 30, f"only {len(scripts)} scripts classified; the guard has drifted"
    assert helpers, "no helper modules found; the skip branch is never exercised"
