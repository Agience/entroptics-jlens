"""The quantisation numbers the README quotes, recomputed from the model's own weights.

`experiments/exp49_quantization_damage.py` is the script; these are its published figures.

The recomputation needs gpt2's weights. It runs whenever they are already in the local HuggingFace
cache and skips when they are not -- a statement about what is on the machine, not about how long
the test takes. Nothing here is gated on being slow: a check that does not run verifies nothing.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest

import entroptics_jlens as je

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SCRIPT = ROOT / "research/experiments/exp49_quantization_damage.py"
# The sdist ships this suite but not every repository file it reads. A shipped test that fails
# for an absent repository file reads as a broken package rather than a file that was never
# part of the distribution, so the module skips instead.
if not SCRIPT.exists():                                          # pragma: no cover - sdist
    pytest.skip("experiments/ is a repository directory and is not shipped",
                allow_module_level=True)
def _gpt2_is_cached() -> bool:
    """Is gpt2 already downloaded? A skip on a missing artifact is honest; a skip on 'this is
    slow' is a test that never runs."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return any(root.rglob("models--gpt2/snapshots/*/*.safetensors")) or         any(root.rglob("models--gpt2/snapshots/*/pytorch_model.bin"))


def quant_rows() -> dict[int, dict[int, float]]:
    """The README's quantisation table, parsed back into numbers."""
    text = README.read_text(encoding="utf-8")
    block = re.search(r"\| layer \| int8 \| int4 \| int3 \|.*?\n\n", text, re.DOTALL)
    assert block, "the quantisation table is no longer in the README in this shape"
    out: dict[int, dict[int, float]] = {}
    for line in block.group(0).splitlines():
        m = re.match(r"\|\s*\*{0,2}(\d+)\*{0,2}\s*\|(.+)\|$", line)
        if not m:
            continue
        vals = [float(c.replace("*", "").strip())
                for c in m.group(2).split("|") if c.replace("*", "").strip()]
        if len(vals) == 3:
            out[int(m.group(1))] = dict(zip((8, 4, 3), vals))
    return out


def test_the_quantisation_table_was_parsed():
    """The premise: a regex matching nothing would make everything below vacuous."""
    rows = quant_rows()
    assert set(rows) == {0, 3, 6, 9, 11}, f"parsed layers {sorted(rows)}"
    assert all(set(v) == {8, 4, 3} for v in rows.values())


def test_the_table_says_what_the_readme_claims_it_says():
    """The two claims the section makes, as opposed to any one cell: int8 holds everywhere and is
    worst at layer 11, and int4 damage is uneven rather than uniform."""
    rows = quant_rows()
    assert all(v[8] > 0.95 for v in rows.values()), "int8 must hold everywhere"
    assert min(rows, key=lambda layer: rows[layer][8]) == 11, "layer 11 is int8's worst"
    int4 = [v[4] for v in rows.values()]
    assert max(int4) - min(int4) > 0.4, "the section's point is that int4 damage is uneven"


def test_fewer_bits_never_read_as_less_damage():
    """Monotonicity, which the table would violate if a row were mistyped."""
    for layer, v in quant_rows().items():
        assert v[8] >= v[4] >= v[3], f"layer {layer} is not monotone in bit width: {v}"


def _script():
    spec = importlib.util.spec_from_file_location("exp49", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_scripts_quantiser_is_a_real_one():
    """Cheap, and worth pinning: a `quantize` that returned its input unchanged would make every
    agreement score 1.0 and the whole table meaningless."""
    exp49 = _script()
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64))
    for bits in (8, 4, 3):
        q = exp49.quantize(W, bits)
        assert q.shape == W.shape
        assert not np.allclose(q, W), f"int{bits} changed nothing"
        assert len(np.unique(q)) <= 2 ** bits, f"int{bits} produced too many levels"
    assert np.abs(exp49.quantize(W, 8) - W).max() < np.abs(exp49.quantize(W, 3) - W).max()
    with pytest.raises(ValueError, match="identically zero"):
        exp49.quantize(np.zeros((8, 8)), 4)


@pytest.mark.skipif(not _gpt2_is_cached(), reason="gpt2 is not in the local HF cache")
def test_the_quantisation_numbers_reproduce():
    import transformers

    exp49 = _script()
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval().float()
    for layer, expected in quant_rows().items():
        W = model.transformer.h[layer].attn.c_proj.weight.detach().numpy().astype(np.float64)
        for bits, want in expected.items():
            got = float(je.principal_angles(W, exp49.quantize(W, bits), k=64).mean())
            assert round(got, 4) == want, f"layer {layer}, int{bits}: {got:.4f} against {want}"

    # The figure quoted in prose rather than in the table.
    w11 = model.transformer.h[11].attn.c_proj.weight.detach().numpy().astype(np.float64)
    assert round(je.participation_ratio(je.energy_spectrum(w11)), 1) == 27.2
    assert round(je.participation_ratio(je.energy_spectrum(exp49.quantize(w11, 4))), 1) == 4.2
