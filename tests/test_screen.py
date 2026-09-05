"""The sweep: which layers are worth an SVD, decided without one.

`identity_share` is `tr(J)/d` and one Frobenius norm -- O(d^2) against a spectrum's O(d^3), 130 ms
against 3047 ms at d = 2560 -- and it is the quantity that PREDICTS the correction rather than a
proxy for it, which is what makes screening on it sound rather than a heuristic.

The property that matters is that it agrees with `decompose` exactly. A screen that disagreed
would send you to the wrong layers, and cheaply arriving at the wrong answer is worse than
expensively arriving at the right one.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je
from entroptics_jlens.cli import main

ROOT = Path(__file__).resolve().parents[1]


def transport(d=96, alpha=0.0, seed=0):
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.standard_normal((d, 8)))[0]
    v = np.linalg.qr(rng.standard_normal((d, 8)))[0]
    return (u * 5.0) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d) + alpha * np.eye(d)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 6.0])
def test_the_cheap_share_equals_what_decompose_computes(alpha):
    """The whole basis for screening. If these ever diverge the screen is sending callers to the
    wrong layers, and it is doing it quickly."""
    J = transport(alpha=alpha)
    a_cheap, share_cheap = je.identity_share(J)
    dec = je.decompose(J)
    assert a_cheap == pytest.approx(dec.alpha, rel=1e-12)
    assert share_cheap == pytest.approx(dec.removed_energy, rel=1e-12)


def test_the_share_does_no_svd():
    """Asserted by cost rather than by inspection: an O(d^3) call on a matrix this size cannot
    finish in the budget below, so passing means no decomposition happened."""
    import time
    J = transport(d=512, alpha=2.0)
    je.identity_share(J)                                  # warm
    t = time.time()
    for _ in range(5):
        je.identity_share(J)
    cheap = (time.time() - t) / 5

    t = time.time()
    np.linalg.svd(J, compute_uv=False)
    full = time.time() - t
    assert cheap < full / 5, f"screen {cheap*1000:.1f} ms against an SVD's {full*1000:.1f} ms"


def test_a_non_square_transport_is_refused():
    with pytest.raises(ValueError, match="square transport"):
        je.identity_share(np.ones((4, 7)))


def test_the_all_zero_transport_reports_no_identity_rather_than_dividing():
    alpha, share = je.identity_share(np.zeros((8, 8)))
    assert alpha == 0.0 and share == 0.0


class _Lens:
    """The minimal shape `screen` needs, so this runs without a checkpoint."""

    def __init__(self, alphas):
        self.source_layers = list(range(len(alphas)))
        self._alphas = alphas

    def jacobian(self, layer):
        return transport(alpha=self._alphas[layer], seed=layer)


def test_screen_flags_the_layers_whose_identity_dominates():
    lens = _Lens([0.0, 0.2, 1.0, 4.0, 8.0])
    rows = je.screen(lens)
    assert [r["layer"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["identity"] for r in rows] == sorted(r["identity"] for r in rows), (
        "the fixture should have a rising identity share, or the flagging is untested")
    assert rows[0]["worth_decomposing"] is False
    assert rows[-1]["worth_decomposing"] is True


def test_the_threshold_is_a_screening_cut_and_the_share_is_reported():
    """Nothing downstream may depend on 0.4: a caller who wants a different cut has the number."""
    lens = _Lens([0.0, 1.0, 4.0, 8.0])
    strict = je.screen(lens, threshold=0.99)
    loose = je.screen(lens, threshold=0.0)
    assert sum(r["worth_decomposing"] for r in strict) < sum(r["worth_decomposing"] for r in loose)
    assert [r["identity"] for r in strict] == [r["identity"] for r in loose], (
        "the share must not depend on the threshold")


def test_screen_takes_a_subset_of_layers():
    lens = _Lens([0.0, 1.0, 4.0, 8.0])
    assert [r["layer"] for r in je.screen(lens, [1, 3])] == [1, 3]


@pytest.mark.skipif(
    not glob.glob(str(ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/*.pt")),
    reason="no gpt2 lens: entroptics-jlens fetch gpt2")
def test_the_command_agrees_with_a_full_audit_on_which_layers_matter(tmp_path):
    """End to end on a real lens: every layer the screen flags must be one the full read finds
    identity-dominated, and vice versa. That equivalence is the screen's whole promise."""
    lens_path = glob.glob(str(ROOT / "lenses/gpt2-small/jlens/Salesforce-wikitext/*.pt"))[0]
    screened, audited = tmp_path / "s.json", tmp_path / "a.json"
    assert main(["screen", lens_path, "--threshold", "0.5", "--json", str(screened)]) == 0
    assert main(["audit", lens_path, "--json", str(audited)]) == 0

    flagged = {r["layer"] for r in json.loads(screened.read_text(encoding="utf-8"))["layers"]
               if r["worth_decomposing"]}
    dominated = {r["layer"] for r in json.loads(audited.read_text(encoding="utf-8"))["layers"]
                 if r["verdict"] == "identity-dominated"}
    assert flagged == dominated, f"screen said {sorted(flagged)}, audit said {sorted(dominated)}"
