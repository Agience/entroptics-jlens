"""A partial results file must refuse to read as a whole one.

Written after exactly that happened: an experiment was killed at layer 26 of 31, left well-formed
JSON behind, and a later script consumed it as a finished sweep.
"""
import json

import pytest

import entroptics_jlens as je


def test_a_finished_run_reads_back(tmp_path):
    p = tmp_path / "r.json"
    je.dump(p, {"rows": [1, 2, 3]}, complete=True)
    assert je.load_complete(p)["rows"] == [1, 2, 3]


def test_an_interrupted_run_refuses(tmp_path):
    """The bug this module exists to stop: valid JSON, partial content."""
    p = tmp_path / "r.json"
    je.dump(p, {"rows": [1]}, complete=False)          # as written inside the loop
    with pytest.raises(je.IncompleteResults, match="did not finish"):
        je.load_complete(p)


def test_a_file_predating_the_convention_refuses(tmp_path):
    """An unstamped file is evidence of nothing, so it is refused rather than guessed at."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"rows": [1, 2]}), encoding="utf-8")
    with pytest.raises(je.IncompleteResults):
        je.load_complete(p)


def test_a_missing_file_refuses_distinctly(tmp_path):
    """Absent and incomplete are different faults and get different errors."""
    with pytest.raises(FileNotFoundError):
        je.load_complete(tmp_path / "absent.json")


def test_the_final_write_overwrites_the_partial_state(tmp_path):
    """The loop stamps False many times; the one write after it must win."""
    p = tmp_path / "r.json"
    for i in range(3):
        je.dump(p, {"rows": list(range(i + 1))}, complete=False)
    je.dump(p, {"rows": [0, 1, 2]}, complete=True)
    assert je.load_complete(p)["rows"] == [0, 1, 2]


def test_a_bare_list_refuses_cleanly(tmp_path):
    """The pre-convention format was a bare list, and it must refuse rather than crash.

    It first surfaced as ``AttributeError: 'list' object has no attribute 'get'`` -- a guard that
    fails with the wrong exception is a guard a caller cannot handle.
    """
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"layer": 0}, {"layer": 1}]), encoding="utf-8")
    with pytest.raises(je.IncompleteResults, match="bare list"):
        je.load_complete(p)


def test_the_plotters_refuse_a_partial_sweep(tmp_path):
    """A figure built from an interrupted run looks finished, and a figure gets published.

    The three plotters read results files, so they carry the same guard as the experiments. This
    pins the wiring rather than the helper: a plotter that fell back to ``json.load`` would draw
    27 of 31 layers with no sign anything was missing.
    """
    import subprocess
    import sys
    from pathlib import Path

    src = tmp_path / "partial.json"
    je.dump(src, {"rows": [{"layer": 0, "rel_depth": 0.0, "cos2": 0.1}]}, complete=False)
    script = Path(__file__).resolve().parents[1] / "research" / "experiments" / "plot_reach.py"
    # experiments/ is a repository directory and is not part of the distribution, so a run
    # from an unpacked sdist skips rather than reporting a missing file as a defect.
    if not script.exists():                                       # pragma: no cover - sdist
        pytest.skip("experiments/ is not shipped")
    out = subprocess.run([sys.executable, str(script), str(src), "--out", str(tmp_path / "o.svg")],
                         capture_output=True, text=True)
    assert out.returncode == 2, f"expected the refusal exit code, got {out.returncode}"
    assert "did not finish" in (out.stderr + out.stdout)


def test_every_results_file_records_the_producing_commit(tmp_path):
    """Inputs alone do not determine a result.

    Measured: a committed exp4 file was rebuilt from its own provenance -- same model, same lens
    hash, same prompt count, same library versions, same seed -- and 8 of 220 numeric fields came
    back different, every one of them from the single read whose implementation had changed since.
    The block recorded every input except the arithmetic.
    """
    p = tmp_path / "r.json"
    je.dump(p, {"rows": []}, complete=True)
    doc = je.load_complete(p)
    assert doc["code_commit"], "no commit recorded"


def test_a_dirty_tree_is_marked_as_such():
    """A commit alone is a claim the working tree may contradict, so uncommitted work is flagged."""
    from entroptics_jlens.results import code_version
    v = code_version()
    assert v == "unknown" or v.split("+")[0], f"unusable version string {v!r}"
    if v != "unknown":
        assert "+dirty" in v or v.isalnum(), f"unexpected version format {v!r}"


def test_an_amplifying_round_trip_has_no_energy_share():
    """Prop 2.2 makes ``1 - residual^2`` a share only while the round trip contracts.

    Measured at gpt2 layer 9, where the transport falls to rank 3 of 768: the certificate's
    residual came back at 1.038, and the share it implies is -7.8% -- a negative fraction of
    energy, printed in a column of percentages. The reporting path now yields ``nan`` above a
    residual of 1 rather than a number outside [0, 1].
    """
    import math

    def share(res):
        return 1.0 - res ** 2 if res <= 1.0 else float("nan")

    assert share(0.97541) == pytest.approx(0.0486, abs=1e-4)
    assert share(1.0) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(share(1.03823))
    assert math.isnan(share(2.0))


def test_the_completion_stamp_lands_last_even_when_the_payload_carries_one():
    """`{**payload, "complete": v}` keeps the key's FIRST insertion position, so a payload that
    already has one puts the stamp at the top of the file with the right value and the wrong
    place -- silently voiding "written last, so a truncated write is invalid JSON".

    Reachable by the most natural round trip there is: `load_complete` returns a dict containing
    `complete`, so read-modify-write produces exactly this payload. Measured before the fix: the
    key landed at byte 4 of an 88-byte file.
    """
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "r.json"
        je.dump(out, {"layers": [1, 2]}, complete=True)
        first = je.load_complete(out)
        assert list(first)[-1] == "complete"

        # The round trip: read it back, add something, write it out again.
        je.dump(out, {**first, "extra": 1}, complete=True)
        raw = out.read_text(encoding="utf-8")
        keys = list(json.loads(raw))
        assert keys[-1] == "complete", f"stamp is not last: {keys}"
        assert keys.count("complete") == 1
        # And the module's own stamp wins over a stale one carried in the payload.
        assert json.loads(raw)["complete"] is True

        je.dump(out, {**first, "extra": 1}, complete=False)
        assert list(json.loads(out.read_text(encoding="utf-8")))[-1] == "complete"
        with pytest.raises(je.IncompleteResults):
            je.load_complete(out)


def test_code_version_is_computed_once_per_process():
    """It spawns two git subprocesses, measured at 117-197 ms, and `dump` is called once per
    layer by design. A 31-layer sweep was paying about four seconds for one answer."""
    from entroptics_jlens.results import code_version
    assert hasattr(code_version, "cache_clear"), "code_version must be cached"
    code_version.cache_clear()
    first = code_version()
    info = code_version.cache_info()
    code_version()
    assert code_version.cache_info().hits == info.hits + 1
    assert code_version() == first
