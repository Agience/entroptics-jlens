"""Entroptics reads of the Jacobian lens.

**The result this package exists for.** A residual stream ADDS each layer's output to what came
before, so a transport from layer *l* to the final layer carries a copy of the identity by
construction, and it grows with depth. That copy flattens the spectrum; the noise floor is
estimated FROM that spectrum; so the floor rises with the identity and buries the real modes
underneath. Measured across every published lens, at each one's deepest fitted layer:

    reading the transport as it stands, against reading it with the identity removed,
    changes the resolved count by 2x to 21x, in 9 of 10 models, across five families,
    at widths 512 to 4096.

Read qwen3-1.7b's deepest layer as it stands and it resolves 3 directions. It resolves 64. At every
model above 1B here, 66-88% of the deepest transport is that pass-through.

The size of the change tracks the identity share (Spearman +0.773 over eleven files, +0.745 over
ten models), and the one lens where the count barely moves is the one with almost no identity to
remove. A structure-free surrogate reproduces that same association, so `exp55` measures the part
belonging to the transport: a median 1.15x to 1.87x depending on which invariants the surrogate
holds. `exp51_the_claim.py` reproduces the headline table in one command.

**The fix is exact and is one line**::

    M = je.decompose(J).residual        # alpha = tr(J)/d, then J - alpha*I

``alpha = tr(J)/d`` is the orthogonal projection of ``J`` onto ``span(I)`` under the Frobenius
inner product: the unique least-squares coefficient against a basis element the architecture
guarantees is present. No threshold, no null, no fitted constant, no training. There is nothing in
it to fail, which is why it replicates.

**The rest of the package.** ``transport_spectrum`` reads resolved rank against a derived floor;
``spectra`` carries the threshold-free reads; ``principal_angles`` says how far into the spectrum
two fits agree; ``coverage`` measures overlap with a real stream against an analytic chance level;
``workspace`` is the runtime half, a projection at 0.93 us/token. ``entroptics-jlens audit`` prints
the columns above side by side.

**What it does not do.** Six independent attempts to turn one of these reads into a PREDICTION have
failed -- escalation, abstention, probe placement, the complement read, quantisation cost, and
subspace summarisation. The reads are exact about geometry and
no test has yet connected the geometry to a downstream outcome. The claim above survives precisely
because it forecasts nothing: it is a statement about what the matrix is.
"""
from .frames import FrameError, as_frame
from .transport import (NULLS, TransportSpectrum, transport_spectrum,
                        spectrum_under_nulls, resolved_transport, spectrum_for_floor,
                        floor_is_resolvable, FLOOR_PRECISION_MARGIN)
from .lenses import (STREAM, TRANSPORT, LayerScreen, layer_screen, complement,
                     stream_side, transport_side, vocab_side, truncated_pair)
from .controls import (gaussian_null, shuffled_entries, matched_spectrum,
                       haar_orthogonal, frobenius_sigma)
from .coverage import Coverage, coverage, coverage_null_sample
from .results import IncompleteResults, dump, load_complete
from .targets import (centred_cosine, final_norm_weight, prenorm_direction,
                      rms_normalize)
from .decompose import (Decomposition, decompose, identity_share, screen,
                        WORTH_DECOMPOSING)
from .spectra import (energy_spectrum, participation_ratio, shannon_rank,
                      principal_angles, gram_spectrum, residual_gram)
from .nulls import (SURROGATES, SampledFloor, sampled_floor, calibration_report,
                    sign_flip, within_row_shuffle, within_col_shuffle, top_singular,
                    provider as sampled_provider)
from .io import LensFile, LensFormatError, load_lens
from .catalog import CATALOG, fetch
from .workspace import Workspace, workspace
from .bench import Bench, Claim, Report, SealBroken

__all__ = [
    "FrameError", "as_frame",
    "NULLS", "TransportSpectrum", "transport_spectrum", "spectrum_under_nulls",
    "resolved_transport", "spectrum_for_floor", "floor_is_resolvable",
    "FLOOR_PRECISION_MARGIN",
    "STREAM", "TRANSPORT", "LayerScreen", "layer_screen", "complement",
    "stream_side", "transport_side", "vocab_side",
    "gaussian_null", "shuffled_entries", "matched_spectrum", "haar_orthogonal",
    "frobenius_sigma",
    "Coverage", "coverage", "coverage_null_sample", "truncated_pair",
    "centred_cosine", "final_norm_weight", "prenorm_direction", "rms_normalize",
    "IncompleteResults", "dump", "load_complete",
    "Decomposition", "decompose", "identity_share", "screen", "WORTH_DECOMPOSING",
    "energy_spectrum", "participation_ratio", "shannon_rank", "principal_angles",
    "gram_spectrum", "residual_gram",
    "SURROGATES", "SampledFloor", "sampled_floor", "calibration_report",
    "sign_flip", "within_row_shuffle", "within_col_shuffle", "top_singular",
    "sampled_provider",
    "LensFile", "LensFormatError", "load_lens",
    "CATALOG", "fetch",
    "Workspace", "workspace",
    "Bench", "Claim", "Report", "SealBroken",
]
