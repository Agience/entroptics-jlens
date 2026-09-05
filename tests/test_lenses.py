"""The Jacobian lens as a Screen side: what the transport drops, and what stands outside it."""
import numpy as np
import pytest

import entroptics_jlens as je


def rig(rng, d=192, K=8, extra=0, T=300, amp=6.0):
    """A transport with a known K-dimensional row space, and a stream carrying structure both
    inside it and (optionally) in ``extra`` directions orthogonal to it."""
    Q = je.haar_orthogonal(d, rng)
    Vin, Vout = Q[:, :K], Q[:, K:K + max(extra, 1)]
    U = je.haar_orthogonal(d, rng)[:, :K]
    J = (U * np.linspace(140.0, 90.0, K)) @ Vin.T + rng.standard_normal((d, d))
    H = rng.standard_normal((T, K)) @ (Vin * amp).T + 0.5 * rng.standard_normal((T, d))
    if extra:
        H = H + rng.standard_normal((T, extra)) @ (Vout[:, :extra] * amp).T
    return J, H


def test_certify_residual_is_exactly_the_annihilated_component():
    """The design claim: entry(H) = H @ J_K.T and inverse(C) = C @ pinv(J_K).T make the round
    trip H @ (pinv(J_K) J_K).T, the projection onto the resolved row space. So the certificate's
    residual IS the part of the residual stream the transport annihilates -- not an analogy."""
    rng = np.random.default_rng(0)
    J, H = rig(rng)
    ls = je.layer_screen(H, J, layer=0)
    Jp = np.linalg.pinv(ls.J_K)
    expected = np.linalg.norm(H - H @ (Jp @ ls.J_K).T) / np.linalg.norm(H)
    assert ls.screen.lossless(je.TRANSPORT, H) == pytest.approx(expected, rel=1e-10)


def test_complement_counts_structure_planted_outside_J_space():
    """Six directions planted orthogonal to the transport's row space move the complement's
    resolved count by exactly six. This is the read the paper has no number for."""
    rng = np.random.default_rng(0)
    J, H_in = rig(rng, extra=0)
    base = je.complement(je.layer_screen(H_in, J, layer=0)).K_signal

    rng = np.random.default_rng(0)
    J2, H_out = rig(rng, extra=6)
    assert np.array_equal(J, J2)                       # same rig, same transport
    assert je.complement(je.layer_screen(H_out, J2, layer=0)).K_signal == base + 6


def test_a_flat_spectrum_reads_as_pure_noise():
    """A scaled rotation has every singular value equal, which is exactly the null: every
    p-value is 1.0 and nothing resolves at any false-alarm level. The floor estimates its
    per-cell variance from the matrix it is handed, so a maximum-entropy matrix cannot stand
    above its own bulk."""
    rng = np.random.default_rng(1)
    J = je.haar_orthogonal(96, rng) * 1e4
    spec = je.transport_spectrum(J)
    assert np.all(spec.pvalue == 1.0)
    assert spec.K == 0 and not spec.saturated
    assert je.transport_spectrum(J, far=0.99999).K == 0
    with pytest.raises(ValueError, match="resolves 0 modes"):
        je.resolved_transport(J)


def test_saturation_is_a_guard_not_a_reachable_state():
    """``layer_screen`` refuses a saturated transport because its truncation is the identity and
    the complement would be empty. No constructed matrix reaches that state -- concentrating the
    spectrum lifts the self-estimated floor with it -- so the guard is defensive. The property
    itself is asserted directly."""
    from entroptics_jlens.transport import TransportSpectrum
    full = TransportSpectrum(shape=(96, 96), far=0.05, null_name="mp", excess_kurtosis=0.0,
                             singular=np.ones(96),
                             deviate=np.zeros(96), pvalue=np.zeros(96), floor=0.0,
                             K=96, energy_resolved=1.0)
    assert full.saturated
    assert not TransportSpectrum(shape=(96, 96), far=0.05, null_name="mp", excess_kurtosis=0.0,
                             singular=np.ones(96),
                                 deviate=np.zeros(96), pvalue=np.zeros(96), floor=0.0,
                                 K=95, energy_resolved=1.0).saturated


def test_d_model_mismatch_refuses():
    rng = np.random.default_rng(2)
    J, H = rig(rng, d=192)
    with pytest.raises(ValueError, match="d_model"):
        je.layer_screen(H[:, :64], J, layer=3)


def test_vocab_side_is_entry_only_and_lands_on_the_token_sub_basis():
    """The vocabulary is the other basis, so it is the other screen -- and there is no honest
    inverse from logits back to a residual."""
    rng = np.random.default_rng(3)
    J, H = rig(rng, d=192)
    J_K, _ = je.resolved_transport(J)
    keep = np.arange(40)
    side = je.vocab_side(J_K, rng.standard_normal((5000, 192)), keep)
    assert "inverse" not in side
    assert side["entry"](H).shape == (H.shape[0], keep.size)


def test_vocab_side_requires_an_explicit_token_subset():
    rng = np.random.default_rng(4)
    J, _ = rig(rng, d=192)
    J_K, _ = je.resolved_transport(J)
    with pytest.raises(ValueError, match="non-empty index"):
        je.vocab_side(J_K, rng.standard_normal((5000, 192)), np.array([], dtype=int))


def test_rank_none_uses_the_transport_whole():
    """The default. The complement read does not need a truncation -- `uncondensed` restricts to
    the directions the RECEIVER resolves, read from the transported frame against the screen's
    own floor, not from J's rank."""
    rng = np.random.default_rng(10)
    J, H = rig(rng, d=128, K=6)
    ls = je.layer_screen(H, J, layer=0, rank=None)
    assert np.allclose(ls.J_K, je.as_frame(J))
    assert np.linalg.matrix_rank(ls.J_K) == min(ls.J_K.shape)


def test_rank_int_truncates_exactly_there():
    rng = np.random.default_rng(11)
    J, H = rig(rng, d=128, K=6)
    ls = je.layer_screen(H, J, layer=0, rank=9)
    assert np.linalg.matrix_rank(ls.J_K, tol=1e-8 * np.linalg.norm(ls.J_K, 2)) == 9


def test_rank_resolved_matches_the_transport_spectrum():
    rng = np.random.default_rng(12)
    J, H = rig(rng, d=128, K=6)
    ls = je.layer_screen(H, J, layer=0, rank="resolved")
    assert np.linalg.matrix_rank(
        ls.J_K, tol=1e-8 * np.linalg.norm(ls.J_K, 2)) == ls.spectrum.K


def test_full_rank_transport_makes_certify_degenerate():
    """Why `rank=None` is right for the complement and wrong for certify: at full rank the
    pseudo-inverse is ill-conditioned, the round trip returns a residual of order round-off, and
    the screen's per-channel whitening lifts that to unit amplitude and resolves 'modes' in it.
    A certificate read there is measuring arithmetic, not the lens."""
    rng = np.random.default_rng(13)
    J, H = rig(rng, d=128, K=6)
    full = je.layer_screen(H, J, layer=0, rank=None)
    trunc = je.layer_screen(H, J, layer=0, rank=9)
    assert full.screen.lossless(je.TRANSPORT, H) < 1e-3      # numerically ~zero
    assert trunc.screen.lossless(je.TRANSPORT, H) > 0.1      # a real null space


@pytest.mark.parametrize("bad,match", [
    (0, "outside"),
    (10_000, "outside"),
    ("nonsense", "rank must be"),
])
def test_bad_rank_refuses(bad, match):
    rng = np.random.default_rng(14)
    J, H = rig(rng, d=128, K=6)
    with pytest.raises(ValueError, match=match):
        je.layer_screen(H, J, layer=0, rank=bad)


def _rms_readout(d, v, seed=0):
    """A stand-in for a model's readout: RMSNorm with a learned gain, then an output head."""
    rng = np.random.default_rng(seed)
    g = 1.0 + 0.1 * rng.standard_normal(d)
    W = rng.standard_normal((v, d)) / np.sqrt(d)

    def unembed(X):
        X = np.asarray(X, dtype=np.float64)
        Z = X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g
        return Z @ W.T
    return g, W, unembed


def test_vocab_side_is_entry_only_unless_a_readout_is_supplied():
    rng = np.random.default_rng(0)
    d, v = 64, 512
    _, _, un = _rms_readout(d, v)
    J = rng.standard_normal((d, d))
    g, W, _ = _rms_readout(d, v)
    assert "inverse" not in je.vocab_side(J, W, np.arange(200))
    assert "inverse" in je.vocab_side(J, W, np.arange(200), invertible=True)


def test_the_vocab_inverse_recovers_direction_but_not_scale():
    """RMSNorm is scale-invariant, so a residual's length is not recoverable from its output.
    The direction is, and that is the payload for carrying a feature between models."""
    rng = np.random.default_rng(1)
    d, v, T = 64, 512, 40
    g, W, un = _rms_readout(d, v)
    keep = np.arange(v)                                   # |keep| >= d, so the head inverts
    J = np.eye(d)                                         # isolate the readout inversion
    side = je.vocab_side(J, W, keep, norm_gain=g, norm_eps=1e-6, invertible=True)
    H = rng.standard_normal((T, d))
    back = side["inverse"](side["entry"](H))
    u = H / np.linalg.norm(H, axis=1, keepdims=True)
    w = back / np.linalg.norm(back, axis=1, keepdims=True)
    cos = (u * w).sum(1)
    assert cos.min() > 0.999, f"directions not recovered: min cos {cos.min():.4f}"
    # scale is genuinely gone, not merely rescaled by a constant
    ratio = np.linalg.norm(back, axis=1) / np.linalg.norm(H, axis=1)
    assert ratio.std() / ratio.mean() > 1e-3, "scale looks recoverable; the docstring is wrong"


def test_the_vocab_inverse_refuses_an_underdetermined_head():
    """Fewer kept tokens than residual dimensions leaves the residual underdetermined; the
    returned vector would be one of infinitely many."""
    d, v = 64, 512
    g, W, _ = _rms_readout(d, v)
    keep = np.arange(32)                                   # 32 < d = 64
    with pytest.raises(ValueError, match="least-squares inverse needs"):
        je.vocab_side(np.eye(d), W, keep, norm_gain=g, norm_eps=1e-6, invertible=True)


def test_the_linear_vocab_inverse_is_exact():
    """Without the norm the conversion is linear, so the round trip returns the surface itself --
    not merely its direction. The scale loss is a property of RMSNorm, not of the crossing."""
    rng = np.random.default_rng(3)
    d, v, T = 64, 512, 40
    _, W, _ = _rms_readout(d, v)
    side = je.vocab_side(np.eye(d), W, np.arange(v), invertible=True)
    H = rng.standard_normal((T, d))
    back = side["inverse"](side["entry"](H))
    assert np.allclose(back, H, atol=1e-8), f"max err {np.abs(back - H).max():.2e}"


def test_a_norm_gain_without_its_epsilon_is_refused():
    """The RMSNorm epsilon belongs to the receiving model, not to this package.

    A crossing that normalises with a house epsilon is undoing an operation the model never
    performed. The values differ by family -- Qwen 1e-6, Llama 1e-5 -- so there is no safe default,
    and the sweep in `exp29_constant_sweep.py` found this one hardcoded.
    """
    rng = np.random.default_rng(0)
    d, v = 16, 64
    W = rng.standard_normal((v, d))
    keep = np.arange(v)
    g = rng.uniform(0.5, 2.0, d)
    with pytest.raises(ValueError, match="norm_eps"):
        je.vocab_side(np.eye(d), W, keep, norm_gain=g, invertible=True)
    side = je.vocab_side(np.eye(d), W, keep, norm_gain=g, norm_eps=1e-5, invertible=True)
    assert "inverse" in side


def test_the_epsilon_actually_reaches_the_arithmetic():
    """A parameter that is accepted and ignored is worse than one that is refused."""
    rng = np.random.default_rng(1)
    d, v = 8, 40
    W = rng.standard_normal((v, d))
    keep = np.arange(v)
    g = np.ones(d)
    X = rng.standard_normal((5, d)) * 1e-4          # small rows: eps dominates the rms
    a = je.vocab_side(np.eye(d), W, keep, norm_gain=g, norm_eps=1e-12)["entry"](X)
    b = je.vocab_side(np.eye(d), W, keep, norm_gain=g, norm_eps=1e-1)["entry"](X)
    assert not np.allclose(a, b), "norm_eps made no difference to the entry"
