"""The §1.5 claim, recomputed from the data it is made on.

The section used to state where the effect lives as two share ranges either side of a 1.5x change
cut. Those ranges are defined by that cut and move with it -- the lower edge of the overlap runs
from 0.095 to 0.422 as the cut goes 1.1x to 3.0x -- so the ranges describe the cut rather than the
transports. What survives the choice is the rank association between identity share and change,
which needs no threshold on either. Both are recomputed here from the stored sweep so the
published numbers cannot drift from the file they came from.
"""
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sweep_and_depth.json"


def depth_rows():
    if not RESULTS.exists():                                         # pragma: no cover - env
        pytest.skip(f"{RESULTS.name} not present")
    return json.loads(RESULTS.read_text(encoding="utf-8"))["depth"]


def test_the_sweep_covers_the_layers_the_section_claims():
    rows = depth_rows()
    assert len(rows) == 81, f"section 1.5 says 81 layers; the file has {len(rows)}"
    assert len({r["model"] for r in rows}) == 5


def test_identity_share_and_change_rank_together_with_no_cut_anywhere():
    """The published rho, over every layer, with no threshold on either variable."""
    stats = pytest.importorskip("scipy.stats")
    rows = depth_rows()
    r = stats.spearmanr([x["identity"] for x in rows], [x["gain"] for x in rows])
    assert r.statistic == pytest.approx(0.530, abs=0.001), f"published +0.530, got {r.statistic}"
    assert r.pvalue < 1e-6


@pytest.mark.parametrize("cut,above,share_above,share_below", [
    (1.1, 26, (0.095, 0.778), (0.003, 0.281)),
    (1.3, 21, (0.267, 0.778), (0.003, 0.433)),
    (1.5, 19, (0.348, 0.778), (0.003, 0.433)),
    (2.0, 12, (0.422, 0.778), (0.003, 0.478)),
    (3.0, 10, (0.422, 0.778), (0.003, 0.555)),
])
def test_the_published_sweep_of_the_change_cut(cut, above, share_above, share_below):
    """Every row of the §1.5 sensitivity table, recomputed."""
    rows = depth_rows()
    hi = [r["identity"] for r in rows if r["gain"] > cut]
    lo = [r["identity"] for r in rows if r["gain"] <= cut]
    assert len(hi) == above
    assert (min(hi), max(hi)) == pytest.approx(share_above, abs=0.001)
    assert (min(lo), max(lo)) == pytest.approx(share_below, abs=0.001)


def test_no_change_cut_separates_the_two_share_populations():
    """The reason the section quotes no dividing share: at every cut across a wide range the two
    ranges overlap, so a layer inside the overlap is on both sides of any line drawn there."""
    rows = depth_rows()
    for cut in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0):
        hi = [r["identity"] for r in rows if r["gain"] > cut]
        lo = [r["identity"] for r in rows if r["gain"] <= cut]
        assert hi and lo
        assert min(hi) <= max(lo), f"cut {cut}x separated them; the section says none does"


def test_the_overlap_edge_moves_far_enough_to_report_the_cut_not_the_data():
    """The published span: the overlap's lower edge runs 0.095 to 0.422 over the swept cuts."""
    rows = depth_rows()
    edges = []
    for cut in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0):
        hi = [r["identity"] for r in rows if r["gain"] > cut]
        lo = [r["identity"] for r in rows if r["gain"] <= cut]
        edges.append(max(min(hi), min(lo)))
    assert min(edges) == pytest.approx(0.095, abs=0.001)
    assert max(edges) == pytest.approx(0.422, abs=0.001)
