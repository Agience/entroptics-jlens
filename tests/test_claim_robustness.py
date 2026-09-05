"""The claim does not depend on the one free parameter in the read.

`K` is counted against a noise floor at a false-alarm rate the caller chooses. If the 2x-to-21x
change only appeared at the value the claim table uses, it would be an artefact of that choice
rather than a property of the read. Swept across 5.7 orders of magnitude it is 9 of 10 models at
every setting, and the factors barely move.

The sweep bounds the one parameter a caller sets. What the count is actually sensitive to is the
variance estimate the floor is built on.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def far_rows() -> list[tuple[float, int, int, float, float]]:
    """The README's far-sweep table: (far, models_over, models_total, low, high)."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| far \| models over 1\.5.*?\n\n", text, re.DOTALL)
    assert block, "the far-sweep table is no longer in the README in the shape this test reads"
    rows = []
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*([\d.e\-]+)\s*\|\s*(\d+)/(\d+)\s*\|\s*"
                     r"([\d.]+)×\s*–\s*([\d.]+)×\s*\|$", line)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5))))
    return rows


def test_the_sweep_table_was_parsed():
    """The premise. A regex matching nothing would make every assertion below vacuous."""
    rows = far_rows()
    assert len(rows) == 5, f"parsed {len(rows)} rows from the far-sweep table"
    fars = [r[0] for r in rows]
    assert min(fars) <= 1e-6 and max(fars) >= 0.5, f"the sweep no longer spans the range: {fars}"


def test_the_claim_holds_at_every_false_alarm_rate():
    for far, over, total, low, high in far_rows():
        assert (over, total) == (9, 10), f"far={far} gives {over}/{total}, not 9/10"
        assert low < 1.5, f"far={far}: the exception should stay below 1.5x, got {low}"
        assert high > 19.0, f"far={far}: the top of the range fell to {high}"


def test_the_factors_are_stable_and_not_merely_present():
    """9 of 10 at every setting would still be weak if the factors swung wildly between them."""
    highs = [r[4] for r in far_rows()]
    lows = [r[3] for r in far_rows()]
    assert max(highs) - min(highs) < 2.0, f"the top factor is unstable across far: {highs}"
    assert max(lows) - min(lows) < 0.2, f"the exception is unstable across far: {lows}"


@pytest.mark.parametrize("far", [0.5, 0.05, 0.005, 1e-6])
def test_a_real_lens_reproduces_the_sweep(far):
    """Recomputed rather than parsed, on a lens cheap enough to run every time. gpt2 sits in the
    middle of the table at 6.5x-7.6x, so it exercises the claim without being its best case."""
    found = glob.glob(str(ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/*.pt"))
    if not found:
        pytest.skip("no gpt2 lens: entroptics-jlens fetch gpt2")
    lens = je.load_lens(found[0])
    J = lens.jacobian(lens.source_layers[-1])
    M = je.decompose(J).residual
    k_raw = je.transport_spectrum(J, far=far, null="mp",
                                  s=np.linalg.svd(J, compute_uv=False)).K
    k_dec = je.transport_spectrum(M, far=far, null="mp",
                                  s=np.linalg.svd(M, compute_uv=False)).K
    factor = k_dec / k_raw
    assert factor > 1.5, f"gpt2 at far={far}: {k_dec}/{k_raw} = {factor:.1f}x"
    assert 6.0 < factor < 8.0, f"gpt2 at far={far} moved outside its measured band: {factor:.1f}x"


def test_the_identity_share_itself_has_no_free_parameter():
    """The correction is exact and takes no false-alarm rate at all -- `alpha = tr(J)/d`. Only the
    COUNTING of resolved modes involves `far`, which is why the sweep above is about K and not
    about the decomposition. Asserted so nobody later adds a threshold to `decompose`."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 64)) + 3.0 * np.eye(64)
    first = je.decompose(A)
    second = je.decompose(A)
    assert first.alpha == second.alpha == pytest.approx(float(np.trace(A)) / 64)
    assert first.removed_energy == second.removed_energy
