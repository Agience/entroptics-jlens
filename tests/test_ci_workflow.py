"""The CI workflow must only invoke tools the install step actually installs.

A workflow that invokes a tool its install step does not provide stops on "No module named X",
and that is invisible locally because the tool is present in any environment where someone has
been running it by hand. Every `run:` step across every workflow is checked here.

Parsed with regexes rather than a YAML library on purpose: adding PyYAML to `[dev]` to test that
`[dev]` is complete would be its own kind of circular.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
# Kept for the messages, which name a file. Every check below covers all of them: `publish.yml`
# runs build and twine, and an undeclared tool there fails a release rather than a test run.
WORKFLOW = ROOT / ".github/workflows/test.yml"
PYPROJECT = ROOT / "pyproject.toml"
# The sdist ships this suite but not the repository files it reads, so a run from an unpacked
# sdist skips here rather than failing collection.
if not WORKFLOW.exists():                                        # pragma: no cover - sdist
    pytest.skip("the CI workflow is a repository file and is not shipped",
                allow_module_level=True)

#: Invoked as `python -m X` but shipped with the interpreter, so no extra can declare them.
STDLIB_MODULES = {"pip", "venv", "unittest", "json", "http", "compileall", "ensurepip"}


def workflow_run_steps() -> list[str]:
    """Every `run:` step across every workflow in the repository."""
    steps = []
    for wf in WORKFLOWS:
        text = wf.read_text(encoding="utf-8")
        steps += [m.group(1).strip()
                  for m in re.finditer(r"^\s*-\s+run:\s*(.+)$", text, re.MULTILINE)]
    return steps


def declared_packages() -> set[str]:
    """Every distribution named in pyproject's `dependencies` or in an extra.

    Scoped to those two places. Matching every `name = [...]` array in the file would sweep in
    the ruff lint selectors and the keyword list, so `e4`, `e741` and `effective-rank` would
    count as installed distributions and a workflow could invoke `python -m build` on the
    strength of an unrelated array.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    blocks = re.findall(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    extras = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text,
                       re.MULTILINE | re.DOTALL)
    if extras:
        blocks += re.findall(r"=\s*\[(.*?)\]", extras.group(1), re.DOTALL)
    names = set()
    for block in blocks:
        for raw in re.findall(r'"([^"]+)"', block):
            name = re.split(r"[<>=!~ \[]", raw, maxsplit=1)[0].strip()
            if name:
                names.add(name.lower().replace("_", "-"))
    return names


def test_the_workflow_has_run_steps_to_check():
    """The premise. A regex that matched nothing would make every test below vacuous."""
    steps = workflow_run_steps()
    assert len(steps) >= 2, f"parsed {len(steps)} run steps from {WORKFLOW.name}; regex drifted"
    assert any("pytest" in s for s in steps), "the workflow no longer runs the test suite"


def test_pyproject_declarations_were_actually_parsed():
    """The same premise for the other side."""
    declared = declared_packages()
    assert {"numpy", "entroptics", "pytest"} <= declared, f"parsed only {sorted(declared)}"


@pytest.mark.parametrize("step", workflow_run_steps(), ids=lambda s: s.split()[0])
def test_every_tool_the_workflow_runs_is_installed_by_it(step: str):
    """`python -m X` must name a stdlib module or a declared distribution."""
    m = re.match(r"python\s+-m\s+([\w.-]+)", step)
    if not m:
        return
    module = m.group(1).split(".")[0].lower()
    if module in STDLIB_MODULES:
        return
    declared = declared_packages()
    assert module.replace("_", "-") in declared, (
        f"the workflow runs `python -m {module}` but no dependency or extra in pyproject.toml "
        f"declares it, so the install step will not provide it. Declared: {sorted(declared)}")


def test_the_install_step_names_an_extra_that_exists():
    text = PYPROJECT.read_text(encoding="utf-8")
    for step in workflow_run_steps():
        for extra in re.findall(r'pip install[^\n]*\.\[([\w,-]+)\]', step):
            for one in extra.split(","):
                assert re.search(rf"^\s*{re.escape(one)}\s*=\s*\[", text, re.MULTILINE), (
                    f"the workflow installs the extra '{one}', which pyproject.toml does not "
                    f"define")
