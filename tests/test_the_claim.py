"""The claim the README leads with, recomputed from the lens files it names.

    "A Jacobian lens carries a large identity component at depth, and a spectral floor derived
     from the matrix it judges is raised by it. Removing the identity first changes the
     resolved count by 2x to 21x."

Ten models across five families, widths 512 to 4096. This is the result the package exists to
deliver, so it gets the most direct test in the suite: parse the table, read the lenses,
recompute every cell.

Every row runs, skipping only when a lens FILE is absent -- a statement about the checkout
rather than about how long the test takes.
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

#: The directory each row's lens lives in, under `lenses/`. The table names models; the files are
#: on disk under the catalogue's own subdirectory names, and the two differ.
LENS_DIR = {
    "qwen3-1.7b": "qwen3-1.7b", "qwen3-4b": "qwen3-4b", "gemma-3-4b": "gemma-3-4b",
    "qwen3.5-4b": "qwen3.5-4b", "gpt2": "gpt2-small", "qwen3.5-0.8b": "qwen3.5-0.8b",
    "gemma-3-1b": "gemma-3-1b", "llama3.1-8b": "llama3.1-8b",
    "pythia-70m": "pythia-70m-deduped", "gemma-3-270m": "gemma-3-270m",
}


def claim_rows() -> dict[str, dict]:
    """The README's claim table, parsed back into numbers."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| model \| width \| identity share \|.*?\n\n", text, re.DOTALL)
    assert block, "the claim table is no longer in the README in the shape this test reads"
    out = {}
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*([\w.\-]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\*{0,2}(\d+)\*{0,2}"
                     r"\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)×?\*{0,2}\s*\|$", line)
        if m:
            out[m.group(1)] = {"width": int(m.group(2)), "identity": float(m.group(3)),
                               "k_raw": int(m.group(4)), "k_dec": int(m.group(5)),
                               "factor": float(m.group(6))}
    return out


def test_the_claim_table_was_parsed():
    """The premise. A regex matching nothing would make every assertion below vacuous."""
    rows = claim_rows()
    assert len(rows) == 10, f"parsed {len(rows)} rows: {sorted(rows)}"
    assert len(set(rows)) == len(rows), "a model appears twice; the count would double-count it"
    assert set(rows) <= set(LENS_DIR), f"unknown models: {set(rows) - set(LENS_DIR)}"


def test_the_headline_range_matches_the_table():
    """The README says "2x to 21x". If a row moved, the sentence has to move with it."""
    factors = [r["factor"] for r in claim_rows().values()]
    over = [f for f in factors if f > 1.5]
    # 9 of the 10 MODELS in the table. There is an eleventh lens FILE -- a second fit of
    # qwen3.5-4b -- which is not a row here and is a replication rather than an independent
    # model, so the headline counts models.
    assert len(over) == 9, f"the claim says 9 of 10 models; the table shows {len(over)} over 1.5x"
    assert 1.9 <= min(over) <= 2.1, f"the low end of '2x to 21x' is {min(over)}"
    assert 21.0 <= max(factors) <= 21.5, f"the high end of '2x to 21x' is {max(factors)}"


def test_the_arithmetic_of_every_row_is_self_consistent():
    """k_dec / k_raw must be the stated factor, whatever the lenses say."""
    for name, r in claim_rows().items():
        assert round(r["k_dec"] / r["k_raw"], 1) == r["factor"], name


def test_the_mechanism_holds_in_the_table():
    """The size of the change tracks the identity share, and the one lens where it barely moves
    is the one with almost no identity. Without this the table is a coincidence across ten
    models. A structure-free surrogate reproduces the same association, so this is a property of
    the floor rather than of the transports."""
    rows = claim_rows()
    ident = [r["identity"] for r in rows.values()]
    fact = [r["factor"] for r in rows.values()]
    rx, ry = np.argsort(np.argsort(ident)), np.argsort(np.argsort(fact))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    assert rho > 0.6, f"identity share should predict the change; Spearman {rho:+.3f}"

    least = min(rows, key=lambda n: rows[n]["factor"])
    assert rows[least]["identity"] == min(ident), (
        f"the lens with the smallest change ({least}) should be the one with the least "
        f"identity to remove; that is what makes this a mechanism rather than a pattern")


def test_every_model_above_1b_is_majority_identity_at_depth():
    """The README's other headline: 66-88% of the deepest transport is pass-through."""
    big = [r for r in claim_rows().values() if r["width"] >= 1024]
    assert len(big) >= 6
    assert all(r["identity"] > 0.5 for r in big), [r["identity"] for r in big]
    assert 0.65 <= min(r["identity"] for r in big) <= 0.67
    assert 0.87 <= max(r["identity"] for r in big) <= 0.88


@pytest.mark.parametrize("model", sorted(LENS_DIR))
def test_the_row_is_what_the_lens_gives(model):
    """The recomputation, one lens at a time so a partial download still checks what it has."""
    rows = claim_rows()
    if model not in rows:
        pytest.skip(f"{model} is not in the README table")
    found = glob.glob(str(ROOT / f"lenses/{LENS_DIR[model]}/jlens/Salesforce-wikitext/*.pt"))
    # The published row has to be what `entroptics-jlens fetch <model>` gives, so the file
    # compared against it is the one CATALOG names. qwen3.5-4b publishes two fits: the
    # catalogue's is n=1000 and reads 183, the other is n=417 and reads 182.
    canonical = je.CATALOG[model][1] if model in je.CATALOG else None
    found = [f for f in found if Path(f).name == canonical] or found
    if not found:
        pytest.skip(f"no lens for {model}: entroptics-jlens fetch {model}")

    lens = je.load_lens(sorted(found)[0])
    top = lens.source_layers[-1]
    J = lens.jacobian(top)
    dec = je.decompose(J)
    k_raw = je.transport_spectrum(J, null="mp", s=np.linalg.svd(J, compute_uv=False)).K
    k_dec = je.transport_spectrum(
        dec.residual, null="mp", s=np.linalg.svd(dec.residual, compute_uv=False)).K

    want = rows[model]
    assert lens.d_model == want["width"], model
    assert round(dec.removed_energy, 3) == want["identity"], model
    assert k_raw == want["k_raw"], f"{model}: K(J) {k_raw} against {want['k_raw']}"
    assert k_dec == want["k_dec"], f"{model}: K(J-aI) {k_dec} against {want['k_dec']}"
