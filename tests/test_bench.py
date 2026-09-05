"""The bench's guards, each named for the failure on 2026-08-22 that motivated it."""
import numpy as np
import pytest

from entroptics_jlens.bench import Bench, SealBroken


def _bench(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return Bench([{"x": float(v)} for v in rng.normal(size=n)], seed=1, name="t")


def test_the_seal_opens_once():
    """A seal that reopens is a development set, and would not have caught the sign reversal."""
    b = _bench()
    arms = {"base": lambda it: it["x"], "arm": lambda it: it["x"] + 1.0}
    c = b.measure(arms).claim("arm", "base")
    b.verify(c, arms)
    with pytest.raises(SealBroken):
        b.verify(c, arms)


def test_a_bench_without_a_control_is_refused():
    """Every 2026-08-22 null that turned out to be perturbation was read off a table with no
    control arm in it."""
    b = _bench()
    with pytest.raises(ValueError, match="no arm named 'base'"):
        b.measure({"arm": lambda it: it["x"]})


def test_the_halves_are_disjoint_and_cover_the_items():
    b = _bench(n=101)
    seen = [id(x) for x in b.open_items]
    assert len(b) == 101
    assert len(seen) == 50 and len(set(seen)) == 50


def test_a_real_effect_holds_on_the_sealed_half():
    b = _bench(n=400)
    arms = {"base": lambda it: it["x"], "arm": lambda it: it["x"] + 5.0}
    c = b.measure(arms).claim("arm", "base")
    got = b.verify(c, arms)
    assert got["held"] and got["sealed_rate"] == 1.0


def test_a_coin_flip_does_not_hold():
    """The case the module exists for: an arm that differs by noise reads well on one half."""
    rng = np.random.default_rng(7)
    b = _bench(n=400)
    arms = {"base": lambda it: it["x"],
            "arm": lambda it: it["x"] + float(rng.normal()) * 0.01}
    got = b.verify(b.measure(arms).claim("arm", "base", margin=0.10), arms)
    assert not got["held"]


def test_the_rate_is_a_share_not_a_mean():
    """A mean halved between disjoint samples where the rate came back identical, so `claim`
    records the rate."""
    b = _bench(n=200)
    arms = {"base": lambda it: it["x"], "arm": lambda it: it["x"] + 0.001}
    c = b.measure(arms).claim("arm", "base")
    assert c.rate == 1.0                    # every item moved, however little
    assert not hasattr(c, "delta")          # and no magnitude is carried


def test_arms_returning_different_counts_are_refused():
    b = _bench(n=40)
    calls = {"n": 0}

    def flaky(it):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("boom")
        return it["x"]

    with pytest.raises(RuntimeError):
        b.measure({"base": lambda it: it["x"], "arm": flaky})


def test_verify_reports_the_controls_it_measured():
    """Running a control on the sealed half and not reporting it is how a table with no control
    in it gets written."""
    b = _bench(n=200)
    arms = {"base": lambda it: it["x"],
            "arm": lambda it: it["x"] + 5.0,
            "control": lambda it: it["x"] - 5.0}
    got = b.verify(b.measure(arms).claim("arm", "base"), arms)
    r = got["sealed_rates_vs_base"]
    assert r["arm"] == 1.0 and r["control"] == 0.0 and "base" not in r


def test_verify_shows_sealed_means_without_claiming_them():
    """Refusing to CLAIM a magnitude and refusing to SHOW one are different. A hybrid arm with a
    better mean and a worse rate is a real pattern, and the seal has to be able to report it."""
    b = _bench(n=200)
    arms = {"base": lambda it: it["x"], "arm": lambda it: it["x"] + 2.0}
    got = b.verify(b.measure(arms).claim("arm", "base"), arms)
    m = got["sealed_means"]
    assert set(m) == {"base", "arm"} and m["arm"] > m["base"]
    assert not hasattr(got["claim"], "delta")     # the claim still carries no magnitude
