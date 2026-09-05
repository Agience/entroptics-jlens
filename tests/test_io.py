"""Reading a lens checkpoint: the format jlens writes, and refusal on everything else."""
import numpy as np
import pytest

import entroptics_jlens as je
from entroptics_jlens.io import LensFormatError

torch = pytest.importorskip("torch")


def write_lens(tmp_path, d=32, layers=(4, 5, 6), dtype=None, name="lens.pt"):
    """The structure jlens.JacobianLens.save writes: float16 transports under 'J'."""
    dtype = dtype or torch.float16
    rng = np.random.default_rng(0)
    ckpt = {"J": {l: torch.tensor(rng.standard_normal((d, d)), dtype=dtype) for l in layers},
            "n_prompts": 1000, "source_layers": list(layers), "d_model": d}
    p = tmp_path / name
    torch.save(ckpt, p)
    return p


def test_round_trip_reads_metadata_and_upcasts_the_transport(tmp_path):
    lens = je.load_lens(write_lens(tmp_path))
    assert lens.d_model == 32 and lens.n_prompts == 1000
    assert lens.source_layers == [4, 5, 6] and len(lens) == 3
    J = lens.jacobian(5)
    assert J.shape == (32, 32) and J.dtype == np.float64


def test_missing_file_refuses_without_synthesising_a_transport(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not synthesise"):
        je.load_lens(tmp_path / "absent.pt")


def test_a_fitting_checkpoint_is_not_a_lens(tmp_path):
    """jlens itself separates the two by the presence of 'J'."""
    p = tmp_path / "ckpt.pt"
    torch.save({"cotangents": torch.zeros(4), "n_prompts": 12}, p)
    with pytest.raises(LensFormatError, match="not a saved lens"):
        je.load_lens(p)


def test_absent_layer_names_the_layers_that_are_present(tmp_path):
    lens = je.load_lens(write_lens(tmp_path))
    with pytest.raises(KeyError, match=r"\[4, 5, 6\]"):
        lens.jacobian(99)


def test_non_square_transport_refuses(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"J": {0: torch.zeros((32, 16), dtype=torch.float16)},
                "n_prompts": 1, "source_layers": [0], "d_model": 32}, p)
    with pytest.raises(LensFormatError, match="square in d_model"):
        je.load_lens(p).jacobian(0)


def test_string_keyed_layers_are_accepted(tmp_path):
    """torch checkpoints round-tripped through some tooling come back string-keyed."""
    p = tmp_path / "str.pt"
    rng = np.random.default_rng(1)
    torch.save({"J": {"3": torch.tensor(rng.standard_normal((16, 16)), dtype=torch.float16)},
                "n_prompts": 1, "source_layers": [3], "d_model": 16}, p)
    assert je.load_lens(p).jacobian(3).shape == (16, 16)
