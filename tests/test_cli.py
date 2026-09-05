"""The command line: argument resolution, refusals, and the one summary that was wrong.

The interesting test here is ``test_the_reported_peak_ignores_layers_that_resolve_nothing``. The
first version of ``audit`` took an unrestricted argmax over participation ratio and reported the
emptiest layer in the file as the band, because PR is threshold-free: it measures how flat a
spectrum is, and white noise is the flattest thing there is. The synthetic lens caught it -- the
planted band reads PR 17 and the noise-only layers outside it read 80 -- and this pins it.
"""
import json

import numpy as np
import pytest

from entroptics_jlens.cli import _layers, _read_streams, _runs, main


# ---------------------------------------------------------------- pure argument handling

def test_layers_all_returns_every_fitted_layer():
    assert _layers("all", [0, 2, 4]) == [0, 2, 4]
    assert _layers(None, [0, 2, 4]) == [0, 2, 4]


def test_layers_accepts_a_list_and_a_range():
    assert _layers("0,4", [0, 2, 4]) == [0, 4]
    assert _layers("2-4", [0, 1, 2, 3, 4]) == [2, 3, 4]


def test_layers_refuses_a_layer_the_lens_does_not_carry():
    """Not clamped, not dropped: a run that silently read a subset would report a curve with
    holes in it as if it were the curve."""
    with pytest.raises(ValueError, match=r"names \[9\]"):
        _layers("0,9", [0, 1, 2])


def test_runs_renders_a_scatter_as_runs_not_as_one_span():
    assert _runs([0, 1, 2, 19]) == "0-2, 19"
    assert _runs([5]) == "5"
    assert _runs([]) == "none"


def test_read_streams_refuses_an_array_of_the_wrong_rank(tmp_path):
    p = tmp_path / "flat.npy"
    np.save(p, np.zeros((4, 8)))
    with pytest.raises(ValueError, match=r"n_layers \+ 1, T, d"):
        _read_streams(p)


def test_read_streams_takes_one_prompt_or_many(tmp_path):
    one, many = tmp_path / "one.npy", tmp_path / "many.npz"
    np.save(one, np.zeros((3, 5, 8)))
    np.savez(many, a=np.zeros((3, 5, 8)), b=np.zeros((3, 6, 8)))
    assert len(_read_streams(one)) == 1
    assert len(_read_streams(many)) == 2


# ---------------------------------------------------------------- end to end

torch = pytest.importorskip("torch")


def planted_lens(tmp_path, d=96, layers=12, band=(4, 7), rank=6, seed=0, name="lens.pt"):
    """A lens whose middle layers carry a known rank and whose ends are pure noise.

    The ends are the point: their spectra are flat, so their participation ratio is the largest
    in the file while their resolved rank is zero.
    """
    rng = np.random.default_rng(seed)
    J = {}
    for l in range(layers):
        noise = rng.standard_normal((d, d)) / np.sqrt(d)
        if band[0] <= l <= band[1]:
            u = np.linalg.qr(rng.standard_normal((d, rank)))[0]
            v = np.linalg.qr(rng.standard_normal((d, rank)))[0]
            noise = noise + (u * 8.0) @ v.T
        J[l] = torch.tensor(noise, dtype=torch.float16)
    p = tmp_path / name
    torch.save({"J": J, "n_prompts": 1000, "source_layers": list(range(layers)), "d_model": d}, p)
    return p


def test_audit_reports_the_planted_rank_and_writes_its_table(tmp_path, capsys):
    out = tmp_path / "audit.json"
    assert main(["audit", str(planted_lens(tmp_path)), "--json", str(out)]) == 0
    rows = {r["layer"]: r for r in json.loads(out.read_text(encoding="utf-8"))["layers"]}
    assert all(rows[l]["K_robust"] == 6 for l in (4, 5, 6, 7))
    assert all(rows[l]["K_robust"] == 0 for l in (0, 1, 10, 11))
    assert "resolves nothing" in capsys.readouterr().out


def identity_cored_lens(tmp_path, d=128, rank=8, alpha=6.0, name="ident.pt"):
    """One layer whose transport is a real map plus a large identity, the shape a residual stream
    forces on a Jacobian lens with depth."""
    rng = np.random.default_rng(3)
    u = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    v = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    J = (u * 3.0) @ v.T + rng.standard_normal((d, d)) / np.sqrt(d) + alpha * np.eye(d)
    p = tmp_path / name
    torch.save({"J": {0: torch.tensor(J, dtype=torch.float16)}, "n_prompts": 1000,
                "source_layers": [0], "d_model": d}, p)
    return p


def test_the_resolved_rank_is_read_after_the_identity_comes_off(tmp_path, capsys):
    """The identity flattens the spectrum and lifts the floor estimated from it, burying the weak
    modes underneath. Measured on the real Qwen3.5-4B lens at layer 30: K is 25 on J and 183 on
    J - alpha*I, the same matrix under the same null at the same far. `audit` reported the 25.

    The premise is asserted first: if the two ranks agreed on this fixture, the test would pass
    whichever matrix the tool read.
    """
    out = tmp_path / "ident.json"
    assert main(["audit", str(identity_cored_lens(tmp_path)), "--json", str(out)]) == 0
    row = json.loads(out.read_text(encoding="utf-8"))["layers"][0]

    assert row["identity_energy"] > 0.5, "fixture must be identity-dominated to exercise this"
    assert row["K_mp"] > row["K_J_mp"], (
        "premise broken: the identity is not burying any mode in this fixture, so the test "
        "cannot tell which matrix the rank was read from")
    assert row["verdict"] == "identity-dominated"
    assert row["K_J_mp"] == 0 and row["K_mp"] == 8, (
        "the fixture plants rank 8; the identity should bury all of it on raw J")
    assert "with the identity removed" in capsys.readouterr().out


def test_the_reported_peak_ignores_layers_that_resolve_nothing(tmp_path, capsys):
    """The regression this file exists for. Assert the premise too: without the restriction the
    argmax would land outside the band, so a passing test here is not passing by accident."""
    p = planted_lens(tmp_path)
    assert main(["audit", str(p), "--json", str(tmp_path / "a.json")]) == 0
    rows = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["layers"]

    unrestricted = max(rows, key=lambda r: r["pr_residual"])["layer"]
    assert unrestricted not in range(4, 8), (
        "premise broken: the unrestricted argmax already lands in the band, so this test would "
        "pass whether or not the restriction is applied")

    reported = int(capsys.readouterr().out.split("effective rank peaks at layer ")[1].split()[0])
    assert reported in range(4, 8)


def test_compare_calls_two_draws_of_one_construction_different_maps(tmp_path):
    a = planted_lens(tmp_path, seed=0, name="a.pt")
    b = planted_lens(tmp_path, seed=7, name="b.pt")
    out = tmp_path / "cmp.json"
    assert main(["compare", str(a), str(b), "--k", "4", "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["layers"]
    assert all(r["verdict"] != "same map" for r in rows)


def test_compare_against_itself_is_the_same_map(tmp_path):
    """The positive control. Without it, "different map" everywhere could be the read failing."""
    p = planted_lens(tmp_path)
    out = tmp_path / "self.json"
    assert main(["compare", str(p), str(p), "--k", "4", "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["layers"]
    assert all(r["verdict"] == "same map" for r in rows)


def test_the_verdict_reads_the_mean_because_the_minimum_fails_on_honest_fits():
    """The measured populations the SAME_MAP boundary sits between.

    Two published fits of one model (Qwen3.5-4B n=417 and n=1000) read these mean canonical
    cosines over their top 400 directions, and cos_min 0.0004 at layer 0 -- so a verdict on the
    minimum would call two fits of one model different maps. Two unrelated maps of the same
    construction read the second row. Recorded here so the boundary cannot drift without the
    evidence for it moving too.
    """
    from entroptics_jlens.cli import DRIFTED, SAME_MAP

    same_model = [0.8587, 0.9909, 0.9960, 0.9985]          # layers 0, 12, 26, 30
    unrelated = [0.1657, 0.1908]                            # test_compare_calls_two_draws...
    assert all(v > SAME_MAP for v in same_model)
    assert all(v < DRIFTED for v in unrelated)
    assert max(unrelated) < DRIFTED < SAME_MAP < min(same_model)


def test_compare_refuses_two_widths(tmp_path, capsys):
    a = planted_lens(tmp_path, d=96, name="a.pt")
    b = planted_lens(tmp_path, d=64, name="b.pt")
    assert main(["compare", str(a), str(b)]) == 2
    assert "different width" in capsys.readouterr().err


def test_coverage_runs_the_random_map_control_beside_the_real_read(tmp_path):
    d, layers = 96, 12
    p = planted_lens(tmp_path, d=d, layers=layers)
    rng = np.random.default_rng(1)
    streams = tmp_path / "s.npz"
    # A structured stream, not white noise: a frame that resolves nothing has no subspace to
    # overlap, and the read correctly reports that rather than a number.
    codes = rng.standard_normal((layers + 1, 64, 8))
    basis = np.linalg.qr(rng.standard_normal((d, 8)))[0]
    np.savez(streams, a=codes @ basis.T * 6.0 + rng.standard_normal((layers + 1, 64, d)))
    out = tmp_path / "cov.json"
    assert main(["coverage", str(p), "--streams", str(streams), "--layers", "4,5",
                 "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["layers"]
    assert all({"coverage", "chance", "random_map", "random_ratio"} <= set(r) for r in rows)
    assert all(r["read"] for r in rows), "the fixture stream must resolve something"
    # The control is a map of the same rank on the same frame; it has to sit at chance, or the
    # read is measuring the frame rather than the transport.
    assert all(0.5 < r["random_ratio"] < 2.0 for r in rows)


def test_coverage_reports_an_unresolvable_frame_rather_than_printing_nan(tmp_path, capsys):
    """White noise resolves nothing, so there is no subspace to overlap. Earlier this divided by
    a zero chance level and printed `nan` under a real heading."""
    d, layers = 96, 12
    p = planted_lens(tmp_path, d=d, layers=layers)
    streams = tmp_path / "s.npz"
    np.savez(streams, a=np.random.default_rng(1).standard_normal((layers + 1, 64, d)))
    assert main(["coverage", str(p), "--streams", str(streams), "--layers", "4,5"]) == 0
    out = capsys.readouterr().out
    assert "nothing resolved" in out and "nan" not in out


def test_coverage_says_when_the_stream_is_too_short_to_resolve_anything(tmp_path, capsys):
    """A (T, d) frame has rank at most T. A 3-token prompt on gpt2 reads k_signal = 1 and reports
    a one-dimensional overlap at several hundred times chance -- a spectacular number that is a
    fact about the prompt. The tool has to say so; the read still runs."""
    d, layers = 96, 12
    p = planted_lens(tmp_path, d=d, layers=layers)
    streams = tmp_path / "short.npz"
    np.savez(streams, a=np.random.default_rng(2).standard_normal((layers + 1, 3, d)))
    assert main(["coverage", str(p), "--streams", str(streams), "--layers", "4"]) == 0
    out = capsys.readouterr().out
    assert "fewer tokens than d=96" in out
    assert "rank at most T" in out


def test_coverage_is_silent_about_length_when_the_streams_are_long_enough(tmp_path, capsys):
    """The negative case. Without it, the note above could be printed unconditionally."""
    d, layers = 96, 12
    p = planted_lens(tmp_path, d=d, layers=layers)
    streams = tmp_path / "long.npz"
    np.savez(streams, a=np.random.default_rng(2).standard_normal((layers + 1, 128, d)))
    assert main(["coverage", str(p), "--streams", str(streams), "--layers", "4"]) == 0
    assert "fewer tokens" not in capsys.readouterr().out


def test_coverage_refuses_streams_of_the_wrong_width(tmp_path, capsys):
    p = planted_lens(tmp_path, d=96, layers=12)
    streams = tmp_path / "s.npz"
    np.savez(streams, a=np.zeros((13, 16, 64)))
    assert main(["coverage", str(p), "--streams", str(streams)]) == 2
    assert "one basis" in capsys.readouterr().err


def test_coverage_refuses_streams_that_are_too_shallow(tmp_path, capsys):
    """Layer l pairs with hidden state l+1. Reading s[l] instead still produces a plausible
    curve, which is why the depth is checked rather than assumed."""
    p = planted_lens(tmp_path, d=96, layers=12)
    streams = tmp_path / "s.npz"
    np.savez(streams, a=np.zeros((6, 16, 96)))
    assert main(["coverage", str(p), "--streams", str(streams)]) == 2
    assert "hidden state index l+1" in capsys.readouterr().err


def test_a_fitting_checkpoint_is_refused_rather_than_read(tmp_path, capsys):
    p = tmp_path / "ckpt.pt"
    torch.save({"cotangents": torch.zeros(4)}, p)
    assert main(["audit", str(p)]) == 2
    assert "not a saved lens" in capsys.readouterr().err


def test_catalog_lists_every_published_lens(capsys):
    from entroptics_jlens.catalog import CATALOG
    assert main(["catalog"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in CATALOG)


def test_fetch_refuses_an_unknown_key(tmp_path, capsys):
    assert main(["fetch", "gpt5", "--dir", str(tmp_path)]) == 2
    assert "unknown lens key" in capsys.readouterr().err


def test_fetch_checks_every_key_before_downloading_any(tmp_path, capsys):
    """A typo in the last of three names must not be found after two multi-gigabyte downloads."""
    assert main(["fetch", "gpt2", "nope", "--dir", str(tmp_path)]) == 2
    out = capsys.readouterr()
    assert "nope" in out.err
    assert "fetching gpt2" not in out.out
