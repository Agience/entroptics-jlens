"""The boundary conversion, and the one input it used to accept silently.

Every read in this package is spectral, so the frame that reaches them has to be the frame the
caller meant. `as_frame` refuses a non-2-D shape, an empty axis and a non-finite cell -- and, as
of 2026-09-02, a complex one, which it previously cast to float64 while discarding the imaginary
part, raising only a numpy ComplexWarning that nothing here turns into an error.
"""
import numpy as np
import pytest

import entroptics_jlens as je
from entroptics_jlens.frames import FrameError


def test_a_complex_frame_is_refused_rather_than_silently_made_real():
    """The spectrum of the real part is not the spectrum of the matrix. Before the refusal this
    returned an answer, for a different matrix, with no error and no visible warning."""
    z = np.ones((4, 4), dtype=np.complex128) * (1.0 + 2.0j)
    with pytest.raises(FrameError, match="complex input"):
        je.as_frame(z, name="X")


def test_the_refusal_names_how_large_the_discarded_part_was():
    """A caller whose imaginary part is round-off wants to know that; one whose imaginary part is
    the signal wants to know that more."""
    z = np.ones((3, 3), dtype=np.complex128) * (1.0 + 7.5j)
    with pytest.raises(FrameError, match="7.5"):
        je.as_frame(z, name="X")


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32"])
def test_every_real_dtype_upcasts_to_float64(dtype):
    assert je.as_frame(np.ones((3, 3), dtype=dtype)).dtype == np.float64


def test_torch_half_precisions_convert_on_the_torch_side():
    """bfloat16 has no numpy counterpart -- `.numpy()` on one raises "unsupported ScalarType
    BFloat16" -- and a published checkpoint has turned up in bf16 before, so the conversion must
    happen before numpy sees it."""
    torch = pytest.importorskip("torch")
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        assert je.as_frame(torch.ones((2, 2), dtype=dt)).dtype == np.float64


def test_a_complex_torch_tensor_is_refused_by_the_same_check():
    torch = pytest.importorskip("torch")
    with pytest.raises(FrameError, match="complex input"):
        je.as_frame(torch.ones((2, 2), dtype=torch.complex64), name="X")


@pytest.mark.parametrize("bad,match", [
    (np.ones(5), "2-D"),
    (np.zeros((0, 3)), "non-empty"),
    (np.array([[1.0, np.nan], [0.0, 1.0]]), "not finite"),
    (np.array([[1.0, np.inf], [0.0, 1.0]]), "not finite"),
])
def test_the_other_refusals_still_hold(bad, match):
    with pytest.raises(FrameError, match=match):
        je.as_frame(bad, name="X")
