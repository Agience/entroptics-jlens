"""The README's measured tables, recomputed from the lens they claim to come from.

`test_readme_example.py` pins the values beside the library snippet. These are the bigger claims:
the `audit` output the README leads with, and the shuffle-inversion table the nulls section turns
on. Both were correct when written and both have already been stale once -- the audit block
carried a `K(mp)`/`K(rob)` pair read on raw `J` until the columns changed under it, and the
opening said "fifteen seconds" for four passes after the run became twenty.

A README number is a claim like any other. This is the line in the findings that says what it was
measured on, expressed as something that fails.

Skips without the published gpt2 lens (`entroptics-jlens fetch gpt2`) rather than reporting a pass
it did not earn. It is ~35 s: two SVDs a layer for the audit table, plus four more for the shuffle
rows.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LENS = ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"

pytestmark = pytest.mark.skipif(
    not LENS.is_file(), reason="needs the published gpt2 lens: entroptics-jlens fetch gpt2")


def audit_rows() -> list[dict]:
    """The README's `audit` block, parsed back into numbers."""
    text = README.read_text(encoding="utf-8")
    # Anchored on PR(J), not on the leading columns. The `screen` block shares the
    # " layer    alpha  identity" prefix and appears earlier, so a looser pattern matches it
    # instead and parses zero rows -- which is what happened when `screen` was documented.
    block = re.search(r"^ layer    alpha  identity     PR\(J\).*?^```", text,
                      re.MULTILINE | re.DOTALL)
    assert block, "the audit block is no longer in the README in the shape this test reads"
    rows = []
    for line in block.group(0).splitlines()[1:]:
        m = re.match(r"\s*(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                     r"\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\S.*)$", line)
        if m:
            rows.append({"layer": int(m[1]), "alpha": float(m[2]), "identity": float(m[3]),
                         "pr_J": float(m[4]), "pr_M": float(m[5]),
                         "K_J_mp": int(m[6]), "K_J_fence": int(m[7]),
                         "K_M_mp": int(m[8]), "K_M_fence": int(m[9]),
                         "verdict": m[10].strip()})
    return rows


def test_the_audit_block_was_parsed_at_all():
    """The premise: a regex that matched nothing would make the recomputation vacuous."""
    rows = audit_rows()
    assert len(rows) == 11, f"parsed {len(rows)} rows from the README's audit block, expected 11"
    assert [r["layer"] for r in rows] == list(range(11))


def test_every_number_in_the_audit_block_is_what_the_lens_gives():
    lens = je.load_lens(LENS)
    for row in audit_rows():
        J = lens.jacobian(row["layer"])
        d = je.decompose(J)
        s_j = np.linalg.svd(J, compute_uv=False)
        s_m = np.linalg.svd(d.residual, compute_uv=False)
        where = f"README audit row for layer {row['layer']}"

        assert round(d.alpha, 3) == row["alpha"], where
        assert round(d.removed_energy, 3) == row["identity"], where
        assert round(je.participation_ratio(s_j), 1) == row["pr_J"], where
        assert round(je.participation_ratio(s_m), 1) == row["pr_M"], where
        assert je.transport_spectrum(J, null="mp", s=s_j).K == row["K_J_mp"], where
        assert je.transport_spectrum(J, null="robust", s=s_j).K == row["K_J_fence"], where
        assert je.transport_spectrum(d.residual, null="mp", s=s_m).K == row["K_M_mp"], where
        assert je.transport_spectrum(d.residual, null="robust",
                                     s=s_m).K == row["K_M_fence"], where


def test_the_prose_claim_about_the_K_columns_still_holds():
    """The README says K(J) collapses with depth and K(J-aI) does not. That is the section's
    whole argument, and it is a claim about the SHAPE of two columns rather than any one cell."""
    rows = audit_rows()
    k_raw = [r["K_J_mp"] for r in rows]
    k_dec = [r["K_M_mp"] for r in rows]
    assert k_raw[0] / k_raw[-1] > 10, f"K(J) no longer collapses: {k_raw}"
    assert k_dec[0] / k_dec[-1] < 2.5, f"K(J-aI) now collapses too: {k_dec}"


def test_the_shuffle_inversion_table_is_what_the_lens_gives():
    """The nulls section's table: four layers where the real transport reads below its own
    entry-shuffle on J, and three of the four stop doing so on J - alpha*I."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| layer \| K\(J\) \| K\(shuffled J\).*?\n\n", text, re.DOTALL)
    assert block, "the shuffle table is no longer in the README in the shape this test reads"
    rows = re.findall(r"^\| (\d+) \| (\d+) \| (\d+) \| \| (\d+) \| ([\d–\-]+) \|$",
                      block.group(0), re.MULTILINE)
    assert len(rows) == 4, f"parsed {len(rows)} shuffle rows, expected 4"

    lens = je.load_lens(LENS)
    for layer, k_j, k_shuf_j, k_m, _k_shuf_m in rows:
        J = lens.jacobian(int(layer))
        M = je.decompose(J).residual
        rng = np.random.default_rng(0)
        assert je.transport_spectrum(J, null="mp").K == int(k_j), f"K(J) at layer {layer}"
        assert je.transport_spectrum(je.shuffled_entries(J, rng),
                                     null="mp").K == int(k_shuf_j), f"K(shuf J) at {layer}"
        assert je.transport_spectrum(M, null="mp").K == int(k_m), f"K(J-aI) at layer {layer}"

    # And the claim the table is making, not just its cells.
    inverted_raw = [r for r in rows if int(r[1]) < int(r[2])]
    assert len(inverted_raw) == 4, "the README says all four rows invert on raw J"


# ---------------------------------------------------------------- the Qwen tables

QWEN = ROOT / "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


def qwen_table() -> dict[str, list[float]]:
    """The README's Qwen identity table, parsed back into numbers."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| \| L0 \| L6 \|.*?\n\n", text, re.DOTALL)
    assert block, "the Qwen identity table is no longer in the README in this shape"
    out = {}
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*\*{0,2}([\w()\-− ]*?)\*{0,2}\s*\|(.+)\|$", line)
        if m and m[1].strip() and not set(m[2]) <= set("-| "):
            cells = [c.replace("*", "").strip() for c in m[2].split("|")]
            try:
                out[m[1].strip()] = [float(c) for c in cells if c]
            except ValueError:
                pass
    return out


def test_the_qwen_table_says_what_the_readme_claims_it_says():
    """The claims, checkable without touching a 406 MB lens.

    The README's argument is not any one cell: it is that PR(J) climbs to the last layer and never
    turns over while PR(J - alpha*I) peaks in the interior and collapses at both ends, and that the
    identity share grows monotonically with depth. A table can be arithmetically right and make the
    opposite point.
    """
    rows = qwen_table()
    assert {"identity", "PR(J)", "PR(J-aI)"} <= set(rows), f"parsed rows: {sorted(rows)}"
    identity, pr_j, pr_m = rows["identity"], rows["PR(J)"], rows["PR(J-aI)"]
    assert len(identity) == len(pr_j) == len(pr_m) == 8

    assert all(b > a for a, b in zip(identity, identity[1:])), "identity must grow with depth"
    assert identity[-1] > 0.75, "the README's headline is that it passes three quarters"
    assert all(b > a for a, b in zip(pr_j, pr_j[1:])), "PR(J) must climb monotonically"
    peak = pr_m.index(max(pr_m))
    assert 0 < peak < len(pr_m) - 1, f"PR(J-aI) must peak in the interior, peaked at {peak}"
    assert pr_m[-1] < pr_m[peak] / 2, "and collapse at the top"


@pytest.mark.skipif(not QWEN.is_file(), reason="fetch qwen3.5-4b to check its table")
def test_every_number_in_the_qwen_table_is_what_the_lens_gives():
    lens = je.load_lens(QWEN)
    rows = qwen_table()
    for i, layer in enumerate((0, 6, 12, 18, 24, 26, 28, 30)):
        J = lens.jacobian(layer)
        d = je.decompose(J)
        where = f"README Qwen table, layer {layer}"
        assert round(d.removed_energy, 3) == rows["identity"][i], where
        assert round(je.participation_ratio(je.energy_spectrum(J)), 1) == rows["PR(J)"][i], where
        assert round(je.participation_ratio(je.energy_spectrum(d.residual)),
                     1) == rows["PR(J-aI)"][i], where
