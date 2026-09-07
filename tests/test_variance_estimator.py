"""The §1.4 variance-estimator sweep, recomputed from the run that produced it.

The `mp` floor is ``sqrt(sigma^2 * (mu + q_TW * sigma_J))``. The Johnstone centring is fixed by the
shape and the Tracy-Widom quantile by the false-alarm rate, which is swept elsewhere, so `sigma^2`
is the last free input -- and it is a de-biased MEDIAN row energy, a choice. `exp56` puts that
choice inside the quantile family and reads the whole claim under each member.

Two things are pinned here. The published table's numbers, so they cannot drift from the file they
came from. And the property that makes the claim independent of the choice: `alpha*I` adds
`alpha^2` to every row's energy, so it moves the entire distribution of row energies rather than a
tail of it, and no location statistic can reject a constant added to all of its inputs. That last
one is checked directly on a matrix rather than taken from the stored run.
"""
import json
from pathlib import Path

import numpy as np
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "variance_estimator.json"

#: (estimator, models over 1.5x, factor low, factor high) as published in §1.4 and the README.
PUBLISHED = [
    ("p=0.10", 10, 1.66, 40.33),
    ("p=0.25", 9, 1.25, 33.67),
    ("p=0.40", 9, 1.06, 26.67),
    ("p=0.50", 9, 1.08, 21.33),
    ("p=0.60", 9, 1.03, 20.00),
    ("p=0.75", 9, 1.00, 16.57),
    ("p=0.90", 9, 1.04, 9.43),
    ("mean", 8, 1.00, 6.86),
]


def sweep():
    if not RESULTS.exists():                                         # pragma: no cover - env
        pytest.skip(f"{RESULTS.name} not present; run exp56_variance_estimator.py")
    d = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not d.get("complete"):                                        # pragma: no cover - env
        pytest.skip("the sweep did not finish")
    return d


def test_the_sweep_covers_the_files_the_section_claims():
    d = sweep()
    assert len(d["lenses"]) == 11, "section 1.4 reads eleven lens files"
    assert d["far"] == 0.05


@pytest.mark.parametrize("label,over,lo,hi", PUBLISHED)
def test_every_published_row_of_the_estimator_sweep(label, over, lo, hi):
    d = sweep()
    rows = d["lenses"]
    assert d["summary"][label]["models_over_1_5x"] == over
    assert d["summary"][label]["n_models"] == 10
    gains = [r["gain"][label] for r in rows]
    assert min(gains) == pytest.approx(lo, abs=0.005)
    assert max(gains) == pytest.approx(hi, abs=0.005)


def test_nine_of_ten_models_hold_across_the_whole_quantile_family():
    """The claim the sweep is for: it does not turn on which location statistic reads the rows."""
    d = sweep()
    quantiles = [lab for lab in d["estimators"] if lab.startswith("p=")]
    assert len(quantiles) == 7
    assert all(d["summary"][lab]["models_over_1_5x"] >= 9 for lab in quantiles), {
        lab: d["summary"][lab]["models_over_1_5x"] for lab in quantiles}
    rhos = [d["summary"][lab]["spearman"] for lab in quantiles]
    assert min(rhos) > 0.5, f"the rank association weakens to {min(rhos):+.3f} somewhere"


def test_the_published_median_is_not_the_choice_that_maximises_the_effect():
    """Evidence against tuning: at p=0.10 every one of the eleven files reads a LARGER factor
    than at the published median, and 10 of 10 models clear 1.5x rather than 9."""
    d = sweep()
    bigger = [r["model"] for r in d["lenses"] if r["gain"]["p=0.10"] > r["gain"]["p=0.50"]]
    assert len(bigger) == 11, f"only {len(bigger)} of 11 files read larger at p=0.10"
    assert d["summary"]["p=0.10"]["models_over_1_5x"] == 10
    assert d["summary"]["p=0.50"]["models_over_1_5x"] == 9


def test_the_mean_is_degenerate_rather_than_a_competing_answer():
    """A mean row energy is dominated by the signal it is meant to be a null for. The row is
    reported because a sweep reports its ends, not because it is a second reading."""
    d = sweep()
    gpt2 = next(r for r in d["lenses"] if r["model"].startswith("gpt2"))
    assert gpt2["K_raw"]["mean"] == gpt2["K_decomposed"]["mean"] == 3
    assert gpt2["K_decomposed"]["p=0.50"] == 39, "the median resolves 39 where the mean finds 3"


def test_the_identity_moves_every_row_so_no_quantile_escapes_it():
    """The mechanism, checked on a matrix rather than read from the run.

    ``alpha*I`` adds ``alpha^2`` to the energy of every row, so EVERY quantile of the row-energy
    distribution shifts by the same amount. This is why the sweep above comes out flat: there is
    no location statistic that sees the identity as an outlier, because it is not in a tail.
    """
    rng = np.random.default_rng(0)
    d = 256
    M = rng.standard_normal((d, d)) / np.sqrt(d)
    alpha = 0.7
    J = M + alpha * np.eye(d)

    e_m = np.einsum("ij,ij->i", M, M)
    e_j = np.einsum("ij,ij->i", J, J)
    # The cross term is 2*alpha*diag(M), zero-mean and small beside alpha^2 = 0.49.
    for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        shift = np.quantile(e_j, p) - np.quantile(e_m, p)
        assert shift == pytest.approx(alpha * alpha, abs=0.12), (
            f"quantile {p} moved by {shift:.3f}, not by alpha^2 = {alpha * alpha:.3f}")

    # And with the cross term gone, it is exact at every quantile.
    e_exact = np.einsum("ij,ij->i", M, M) + alpha * alpha
    for p in (0.05, 0.5, 0.95):
        assert np.quantile(e_exact, p) - np.quantile(e_m, p) == pytest.approx(alpha * alpha,
                                                                             abs=1e-12)


def test_a_perturbation_of_the_same_energy_that_a_quantile_does_locate():
    """The contrast that gives the test above its teeth, and the boundary is exact.

    Without this, "every quantile shifts by the perturbation's energy" could be a property of any
    perturbation rather than of the identity. Take a spike carrying the SAME Frobenius energy as
    ``alpha*I`` but concentrated in 8 of 256 rows. A quantile then splits into two regimes at
    ``1 - 8/256 = 0.9688``: below it the estimator reads untouched rows and barely moves, above it
    the estimator reads the spiked rows and moves by an order of magnitude more than the identity
    ever does.

    That split is what "a robust estimator can reject it" means -- the perturbation lives in a
    tail, so there are quantiles that see it and quantiles that do not. The identity has no such
    boundary. It is in every quantile, which is why the sweep above is flat.
    """
    rng = np.random.default_rng(0)
    d, alpha, n_spiked = 256, 0.7, 8
    M = rng.standard_normal((d, d)) / np.sqrt(d)
    e_m = np.einsum("ij,ij->i", M, M)

    S = np.zeros((d, d))
    S[rng.choice(d, size=n_spiked, replace=False)] = rng.standard_normal((n_spiked, d))
    S *= np.sqrt(alpha * alpha * d / float((S ** 2).sum()))
    assert float((S ** 2).sum()) == pytest.approx(alpha * alpha * d, rel=1e-9), (
        "the spike must carry the identity's energy for the comparison to mean anything")

    e_s = np.einsum("ij,ij->i", M + S, M + S)
    boundary = 1.0 - n_spiked / d

    # Below the boundary the estimator does not see it: under 5% of what the identity moves.
    for p in (0.05, 0.25, 0.5, 0.75, 0.90):
        assert p < boundary
        shift = np.quantile(e_s, p) - np.quantile(e_m, p)
        assert abs(shift) < 0.06 * alpha * alpha, (
            f"quantile {p} moved {shift:.4f} under a concentrated spike of the same energy; "
            f"the identity moves every quantile by {alpha * alpha:.4f}")

    # Above it the estimator is reading the spiked rows themselves.
    for p in (0.98, 0.99):
        assert p > boundary
        shift = np.quantile(e_s, p) - np.quantile(e_m, p)
        assert shift > 10.0 * alpha * alpha, (
            f"quantile {p} sits above the spiked fraction and should be reading the spike, "
            f"but moved only {shift:.4f}")
