"""Every experiment script must import cleanly.

Twice in one session an experiment broke on an import rather than on its science: once when a
checkpoint tensor turned out to be bf16 and `safetensors` was opened with the numpy framework,
and once when helpers moved into the package and two scripts still imported them from a sibling
script. Both failed after the expensive part had already started, or would have.

Importing a module runs its top-level statements without running `main`, which is exactly the
part that breaks. This is a cheap standing guard over a growing directory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "research" / "experiments").glob("*.py"))
# The sdist ships this suite but not the repository files it reads, so a run from an unpacked
# sdist skips here rather than failing collection. Without the guard pytest aborts the whole
# run before a single assertion, which reads as a broken package rather than an absent file.
if not EXPERIMENTS:                                              # pragma: no cover - sdist
    pytest.skip("experiments/ is a repository directory and is not shipped",
                allow_module_level=True)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_exp_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_imports(path):
    """No model, no corpus, no GPU -- just the top level."""
    sys.path.insert(0, str(path.parent))
    try:
        _load(path)
    finally:
        sys.path.remove(str(path.parent))


def test_the_directory_is_actually_being_scanned():
    """A glob that silently matches nothing would make every check above vacuous."""
    assert len(EXPERIMENTS) >= 15, f"only {len(EXPERIMENTS)} experiment scripts found"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_has_a_module_docstring(path):
    """Each script is the record of one experiment, so it has to say which."""
    text = path.read_text(encoding="utf-8").lstrip()
    assert text.startswith('"""'), f"{path.name} opens without a docstring"
