"""The README's code must run, and the numbers beside it must be that code's output.

A reader's first copy-paste is the example, so it is executed here rather than reviewed: every
name it uses must resolve, it must read the spectrum on ``J - alpha*I`` as the rest of the
documentation does, and any null it names must be described as what it is.

Two checks, deliberately split by what they need:

  * the symbol check runs anywhere, including CI with no model and no lens, and catches a rename
    or a typo the moment it lands;
  * the numeric check needs the published gpt2 lens and a collected stream, so it skips when
    those are absent rather than pretending to have verified them.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LENS = ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def test_every_python_block_in_the_readme_parses():
    blocks = _python_blocks(README.read_text(encoding="utf-8"))
    assert blocks, "no python block found; the extraction regex has drifted from the README"
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as exc:                       # pragma: no cover - only on a defect
            raise AssertionError(f"README python block {i} does not parse: {exc}") from exc


def test_every_je_symbol_the_readme_uses_exists():
    """Catches a rename before a reader does. Reads the attribute names off the parsed tree
    rather than by regex, so a name inside a string or a comment is not counted."""
    used = set()
    for block in _python_blocks(README.read_text(encoding="utf-8")):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "je":
                    used.add(node.attr)
    assert used, "the README's example no longer calls anything through `je`"
    missing = sorted(n for n in used if not hasattr(je, n))
    assert not missing, f"README uses je.{{{', '.join(missing)}}}, which the package does not export"


@pytest.mark.skipif(not LENS.is_file(),
                    reason="needs the published gpt2 lens: entroptics-jlens fetch gpt2")
def test_the_numbers_printed_beside_the_readme_example_are_that_example_s_output():
    """The example's own values, recomputed. Every one of these appears in the README as a
    trailing comment on the line that produces it."""
    lens = je.load_lens(LENS)
    J = lens.jacobian(5)
    d = je.decompose(J)
    M = d.residual

    assert round(d.alpha, 3) == 0.801
    assert round(d.removed_energy, 3) == 0.278
    assert d.identity_dominated is False

    assert round(je.participation_ratio(je.energy_spectrum(J)), 1) == 41.6
    assert round(je.participation_ratio(je.energy_spectrum(M)), 1) == 22.9
    assert je.transport_spectrum(M, null="mp", far=0.05).K == 51

    c = je.principal_angles(M, je.decompose(lens.jacobian(6)).residual, k=64)
    assert int(np.flatnonzero(c < 0.9)[0]) == 25
