"""Where the correction applies, and where it does not.

The claim's table is each model's deepest layer. A reader working at layer 8 of 24 needs to know
whether it reaches them, and the honest answer is that it does not: below relative depth 0.6 the
correction is 1.0x. Stating only the deepest-layer figure would let someone apply it where it does
not hold, or dismiss it because their own middle layer showed nothing.

The cut is the identity share rather than the depth index, which matters because depth is a proxy
and the share is the mechanism. `audit` prints the share, so a reader can check their own layer
instead of inferring it.
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


def depth_rows() -> list[dict]:
    """The README's depth table: (low, high, identity, correction, over, total)."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| relative depth \| median identity.*?\n\n", text, re.DOTALL)
    assert block, "the depth table is no longer in the README in the shape this test reads"
    rows = []
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*\*{0,2}([\d.]+)\s*–\s*([\d.]+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}"
                     r"\s*\|\s*\*{0,2}([\d.]+)×\*{0,2}\s*\|\s*\*{0,2}(\d+)\s*/\s*(\d+)\*{0,2}"
                     r"\s*\|$", line)
        if m:
            rows.append({"low": float(m.group(1)), "high": float(m.group(2)),
                         "identity": float(m.group(3)), "correction": float(m.group(4)),
                         "over": int(m.group(5)), "total": int(m.group(6))})
    return rows


def test_the_depth_table_was_parsed():
    """The premise. A regex matching nothing would make every assertion below vacuous."""
    rows = depth_rows()
    assert len(rows) == 5, f"parsed {len(rows)} depth buckets"
    assert rows[0]["low"] == 0.0 and rows[-1]["high"] == 1.0


def test_the_correction_is_absent_in_the_shallow_half():
    """The scope of the claim, asserted from the same table that states it. If this ever stops
    holding the README's "below 0.6 the correction is nothing" has to change with it."""
    for row in depth_rows():
        if row["high"] <= 0.6:
            assert row["over"] == 0, f"{row['low']}-{row['high']} now has {row['over']} over 1.5x"
            assert row["correction"] < 1.2, row


def test_the_correction_is_large_in_the_deepest_fifth():
    deepest = depth_rows()[-1]
    assert deepest["correction"] > 3.0, deepest
    assert deepest["over"] / deepest["total"] > 0.75, deepest


def test_the_identity_share_grows_with_depth():
    """The mechanism behind the depth pattern: the change tracks the identity, and the identity
    grows with depth. Without this the table is a depth coincidence.

    Growth rather than strict monotonicity, because the measured profile is not monotone: the two
    shallowest buckets read 0.040 and 0.017, a dip that survives because layer 0 carries a larger
    share than layers 1-2 on several models. What the data supports is the rise from the middle of
    the network onward, and an order of magnitude between the ends.
    """
    ident = [r["identity"] for r in depth_rows()]
    assert all(b >= a for a, b in zip(ident[2:], ident[3:])), (
        f"the share should rise from relative depth 0.4 onward: {ident}")
    assert ident[-1] > 10 * ident[0], f"the share should grow by an order of magnitude: {ident}"


@pytest.mark.parametrize("layer_frac,expect_correction", [(0.0, False), (1.0, True)])
def test_a_real_lens_shows_the_pattern(layer_frac, expect_correction):
    """Recomputed on gpt2: its shallowest layer should need no correction and its deepest should."""
    found = glob.glob(str(ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/*.pt"))
    if not found:
        pytest.skip("no gpt2 lens: entroptics-jlens fetch gpt2")
    lens = je.load_lens(found[0])
    layers = lens.source_layers
    layer = layers[0] if layer_frac == 0.0 else layers[-1]

    J = lens.jacobian(layer)
    M = je.decompose(J).residual
    k_raw = je.transport_spectrum(J, null="mp", s=np.linalg.svd(J, compute_uv=False)).K
    k_dec = je.transport_spectrum(M, null="mp", s=np.linalg.svd(M, compute_uv=False)).K
    factor = k_dec / k_raw

    if expect_correction:
        assert factor > 1.5, f"layer {layer}: {factor:.2f}x"
    else:
        assert factor < 1.5, f"layer {layer}: {factor:.2f}x -- the shallow end should need none"
