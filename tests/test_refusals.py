"""The refusals nothing was exercising.

"Nothing here truncates, imputes or substitutes silently" is this package's central claim, and it
is carried by 55 explicit `raise` statements. An audit on 2026-09-02 -- enumerating them by AST
and diffing against what the tests assert -- found roughly half with no test naming their message
at all. A refusal path that is never taken is a branch nobody has run: its message can name the
wrong variable, its condition can be unreachable, and the guarantee is a comment.

These are the ones that were genuinely uncovered. Each asserts the message and not merely the
type, because a refusal whose text is wrong has failed at the only job it has -- telling the
caller what was found.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics_jlens as je


# ---------------------------------------------------------------- decompose

def test_a_non_square_transport_is_refused():
    """The identity component is only defined for a map from a basis to itself."""
    with pytest.raises(ValueError, match="square transport"):
        je.decompose(np.ones((4, 7)))


def test_the_refusal_names_the_shape_it_was_given():
    with pytest.raises(ValueError, match=r"\(4, 7\)"):
        je.decompose(np.ones((4, 7)))


# ---------------------------------------------------------------- coverage

def test_coverage_refuses_two_frames_on_different_bases():
    """A subspace overlap is only defined for two frames on one basis; without this the two
    resolved bases would come from different spaces and the cosines would be meaningless."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="a subspace overlap needs both on one basis"):
        je.coverage(rng.standard_normal((32, 16)), rng.standard_normal((32, 20)))


# ---------------------------------------------------------------- truncated_pair

def test_a_transport_of_all_zeros_is_refused_rather_than_inverted():
    with pytest.raises(ValueError, match="identically zero"):
        je.truncated_pair(np.zeros((16, 16)), 3)


def test_a_rank_at_the_arithmetic_floor_is_refused():
    """Inverting a singular value at the float floor divides by numerical dust; the round trip
    stops contracting and the residual that comes back is the arithmetic, not the transport."""
    rng = np.random.default_rng(1)
    d = 48
    q1, q2 = je.haar_orthogonal(d, rng), je.haar_orthogonal(d, rng)
    A = (q1 * np.logspace(0, -20, d)) @ q2.T
    with pytest.raises(ValueError, match="arithmetic"):
        je.truncated_pair(A, d)


# ---------------------------------------------------------------- io

def test_a_layer_the_lens_does_not_carry_is_refused_with_the_ones_it_does():
    torch = pytest.importorskip("torch")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "lens.pt"
        torch.save({"J": {4: torch.zeros((8, 8), dtype=torch.float16)},
                    "source_layers": [4], "d_model": 8, "n_prompts": 1}, p)
        lens = je.load_lens(p)
        with pytest.raises(KeyError, match="no transport for layer 9"):
            lens.jacobian(9)
        # The message has to carry what IS there, or the caller has to go and look.
        with pytest.raises(KeyError, match=r"\[4\]"):
            lens.jacobian(9)


# ---------------------------------------------------------------- results

def test_a_missing_results_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no results file"):
        je.load_complete(tmp_path / "absent.json")


def test_a_bare_list_predates_the_stamp_and_is_refused(tmp_path):
    """Files written before the completion convention hold a bare list. They carry no evidence
    either way, and guessing is what the module exists to stop."""
    p = tmp_path / "old.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(je.IncompleteResults, match="holds a bare list"):
        je.load_complete(p)


# ---------------------------------------------------------------- bench

def test_a_bench_too_small_to_seal_is_refused():
    with pytest.raises(ValueError, match="not a bench"):
        je.Bench([1, 2, 3])


def test_an_empty_arm_mapping_is_refused_by_name():
    """An empty arm mapping is refused by name. Unequal counts cannot arise -- every arm is
    scored over the same items -- so a count mismatch would be the wrong message for it."""
    bench = je.Bench(list(range(20)))
    with pytest.raises(ValueError, match="no arms to measure"):
        bench._measure(bench.open_items, {})


def test_verify_does_not_burn_the_seal_on_a_call_it_cannot_answer():
    """The ordering that matters. `self._opened = True` used to come before the measurement, so
    a `verify` missing the arm its claim names consumed the one-shot seal and returned an error.

    The assertion that matters is the second one: the seal must still be usable afterwards."""
    bench = je.Bench(list(range(40)))
    rng = np.random.default_rng(0)
    lift = {i: float(rng.normal(0.4, 1.0)) for i in range(40)}
    arms = {"base": lambda i: 0.0, "treat": lambda i: lift[i]}
    claim = bench.measure(arms).claim("treat", "base")

    with pytest.raises(ValueError, match="were not supplied"):
        bench.verify(claim, {"base": lambda i: 0.0})           # no "treat"

    # The seal survived, so the bench is still worth something.
    report = bench.verify(claim, arms)
    assert report["verdict"] in ("HELD", "DID NOT HOLD")
    assert report["sealed_n"] == 20


# ---------------------------------------------------------------- targets

def test_a_zero_gain_entry_is_refused_rather_than_divided_by():
    """A zero in the final norm gain makes that coordinate unrecoverable; dividing would produce
    an infinity and imputing would invent the direction."""
    with pytest.raises(ValueError, match="zero entry"):
        je.prenorm_direction(np.ones((4, 3)), np.array([1.0, 0.0, 1.0]))


def test_a_gain_vector_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="columns and the gain vector"):
        je.rms_normalize(np.ones((4, 8)), np.ones(5))


def test_centred_cosine_refuses_mismatched_frames():
    with pytest.raises(ValueError, match="must match"):
        je.centred_cosine(np.ones((10, 4)), np.ones((10, 5)))
