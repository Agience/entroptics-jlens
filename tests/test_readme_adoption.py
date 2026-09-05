"""The before/after table the README shows a reader adopting the fix.

It is the most load-bearing table in the document for someone deciding whether to bother: it is
the number their own loop currently produces beside the number it would produce after inserting
one line. Wrong there and they either make a change they did not need or skip one they did.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
QWEN = ROOT / "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


def before_after_rows() -> dict[int, tuple[int, int]]:
    """The README's "in your existing loop" table: layer -> (before, after)."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| layer \| before \| after \| off by \|.*?\n\n", text, re.DOTALL)
    assert block, "the before/after table is no longer in the README in this shape"
    rows: dict[int, tuple[int, int]] = {}
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}(\d+)\*{0,2}"
                     r"\s*\|\s*\*{0,2}[\d.]+×\*{0,2}\s*\|$", line)
        if m:
            rows[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return rows


def test_the_table_was_parsed():
    """The premise. A regex matching nothing makes every check below vacuous."""
    rows = before_after_rows()
    assert set(rows) == {0, 12, 24, 30}, f"parsed layers {sorted(rows)}"


def test_the_table_makes_the_point_the_prose_makes():
    """Nothing changes shallow, a lot changes deep. That is the reason a reader would adopt this,
    and a table that no longer showed it would be arguing against the surrounding text."""
    rows = before_after_rows()
    before0, after0 = rows[0]
    before30, after30 = rows[30]
    assert abs(after0 - before0) / before0 < 0.05, "layer 0 should barely move"
    assert after30 / before30 > 5.0, "layer 30 carries the table"


def test_the_shallow_rows_are_not_all_increases():
    """Honesty check on the table itself. Removing a small identity takes real energy out and the
    count can go DOWN -- layer 12 reads 203 then 178. A table showing only increases would be
    selling the correction rather than reporting it."""
    rows = before_after_rows()
    assert any(after < before for before, after in rows.values()), (
        "the table should include a layer where the correction lowers the count, because that "
        "happens and hiding it would misrepresent the effect")


@pytest.mark.skipif(not QWEN.is_file(), reason="fetch qwen3.5-4b to check its table")
def test_the_numbers_are_what_the_lens_gives():
    lens = je.load_lens(QWEN)
    for layer, (before, after) in before_after_rows().items():
        J = lens.jacobian(layer)
        M = je.decompose(J).residual
        k_before = je.transport_spectrum(J, null="mp", s=np.linalg.svd(J, compute_uv=False)).K
        k_after = je.transport_spectrum(M, null="mp", s=np.linalg.svd(M, compute_uv=False)).K
        assert k_before == before, f"layer {layer}: K(J) is {k_before}, README says {before}"
        assert k_after == after, f"layer {layer}: K(J-aI) is {k_after}, README says {after}"


def test_the_readme_states_where_the_entroptics_boundary_is():
    """The README claims `decompose` is numpy and the measurement is entroptics. That is a
    structural claim about this package and it is cheap to keep true."""
    import inspect

    from entroptics_jlens import decompose as decompose_mod
    src = inspect.getsource(decompose_mod)
    assert "entroptics" not in src.replace("entroptics_jlens", ""), (
        "decompose.py now imports entroptics; the README says it does not")

    from entroptics_jlens import transport as transport_mod
    assert "from entroptics" in inspect.getsource(transport_mod), (
        "transport.py no longer uses entroptics; the README says the measurement comes from it")
