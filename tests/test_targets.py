"""The pre-norm recovery is an inversion, so it can be checked exactly.

Both corrections in `targets` were found only after they had silently distorted three experiments.
These pin them so they cannot come back.
"""
import numpy as np
import pytest

import entroptics_jlens as je


def rms_norm(x, w):
    """The operation the model applies, written out so the test does not depend on torch."""
    return (x / np.sqrt((x ** 2).mean(1, keepdims=True))) * w


def test_prenorm_direction_inverts_rms_norm_exactly():
    """Recovery is exact in direction; only the per-token scale is unrecoverable."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128)) * rng.uniform(0.5, 4.0, (64, 1))   # varied token scales
    w = rng.uniform(0.5, 2.0, 128)
    rec = je.prenorm_direction(rms_norm(x, w), w)
    cos = (rec * x).sum(1) / np.sqrt((rec ** 2).sum(1) * (x ** 2).sum(1))
    assert np.allclose(cos, 1.0, atol=1e-10), f"worst cosine {cos.min()}"


def test_prenorm_direction_refuses_a_zero_gain():
    """A coordinate scaled to zero is genuinely gone; a large finite number is not an answer."""
    w = np.ones(8)
    w[3] = 0.0
    with pytest.raises(ValueError, match="not\n?.*recoverable|recoverable"):
        je.prenorm_direction(np.ones((4, 8)), w)


def test_normalising_both_sides_is_what_makes_the_comparison_exact():
    """The subtlety that makes the comparison exact.

    ``prenorm_direction`` returns the pre-norm frame at UNIT rms -- the per-token scale is gone,
    and a token-centred statistic is not invariant under rescaling one side per token. So scoring a
    raw prediction against the recovered direction still compares two differently-scaled frames.

    Applying the readout's own norm to both sides is what removes the scale from both, and it is
    exact: a transport proportional to the truth then scores 1.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((96, 64)) * rng.uniform(0.5, 4.0, (96, 1))
    w = rng.uniform(0.5, 2.0, 64)
    y = rms_norm(x, w)                                   # what the model hands back
    assert je.centred_cosine(je.rms_normalize(3.7 * x, w), y) == pytest.approx(1.0, abs=1e-10)
    assert je.centred_cosine(x, je.prenorm_direction(y, w)) < 1.0    # the mixed comparison is not


def test_rms_normalize_reproduces_the_models_own_norm():
    """It is the readout's operation, so it must equal it entrywise."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((32, 16)) * 3.0
    w = rng.uniform(0.5, 2.0, 16)
    assert np.allclose(je.rms_normalize(x, w), rms_norm(x, w), atol=1e-12)


def test_centred_cosine_ignores_a_shared_offset():
    """Centring is what makes this a measure of the transport and not of the shared component."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((80, 32))
    b = rng.standard_normal((80, 32))
    off = rng.standard_normal(32) * 50.0
    assert je.centred_cosine(a + off, b + off) == pytest.approx(je.centred_cosine(a, b), abs=1e-9)


def test_the_offset_the_centring_removes_would_otherwise_dominate_the_answer():
    """The invariance above is only worth something if the shared component was large enough to
    matter, and that has to be measured rather than assumed -- the same test passes on an offset
    of zero.

    Measured: two INDEPENDENT frames sharing one constant offset score +0.9997 on a plain
    per-token cosine and +0.0127 centred. The choice of scoring is not a detail here; the
    published cos^2 ceiling for the linearisation moves from 0.69 to 0.85 on scoring alone,
    while its floor does not.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 64))
    b = rng.standard_normal((200, 64))
    off = 50.0 * rng.standard_normal(64)

    plain = float(np.mean(((a + off) * (b + off)).sum(1) /
                          (np.linalg.norm(a + off, axis=1) * np.linalg.norm(b + off, axis=1))))
    centred = je.centred_cosine(a + off, b + off)

    assert plain > 0.99, f"premise broken: the offset does not dominate ({plain})"
    assert abs(centred) < 0.05, f"centring did not remove it ({centred})"


def test_centred_cosine_is_pinned_at_both_ends():
    """A scoring function that returned a constant would pass every invariance test above."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 64))
    assert je.centred_cosine(a, a) == pytest.approx(1.0, abs=1e-9)
    assert je.centred_cosine(a, -a) == pytest.approx(-1.0, abs=1e-9)
