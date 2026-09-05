"""The published Jacobian lenses, and how to get one.

A lens is a transport only -- no model weights, no corpus, no optimiser state -- which is what
makes every read in this package cheap. pythia-70m is 2.6 MB; the Qwen3.5-4B fit the walkthrough
uses is 406 MB. Nothing here downloads a model.

The files live in the ``neuronpedia/jacobian-lens`` mirror of the lenses published with the
Jacobian-lens work. Two entries sit on a non-``main`` revision because their ``n=1000`` fits were
published separately from the ``n=417`` ones; the revision is part of the identity of a lens, and
reading the wrong one silently answers a different question (the two Qwen3.5-4B fits differ by
~5% in participation ratio).
"""
from __future__ import annotations

from pathlib import Path

REPO = "neuronpedia/jacobian-lens"
BASE = "jlens/Salesforce-wikitext"

#: short key -> (repo subdirectory, filename, revision, approximate MB)
CATALOG: dict[str, tuple[str, str, str, float]] = {
    "pythia-70m":    ("pythia-70m-deduped", "pythia-70m-deduped_jacobian_lens.pt", "main", 2.6),
    "gpt2":          ("gpt2-small", "gpt2_jacobian_lens.pt", "main", 13.0),
    "gemma-3-270m":  ("gemma-3-270m", "gemma-3-270m_jacobian_lens.pt", "main", 13.9),
    "qwen3.5-0.8b":  ("qwen3.5-0.8b", "Qwen3.5-0.8B_jacobian_lens.pt", "main", 48.2),
    "gemma-3-1b":    ("gemma-3-1b", "gemma-3-1b-pt_jacobian_lens.pt", "main", 66.4),
    "qwen3.5-2b":    ("qwen3.5-2b-pt", "Qwen3.5-2B-Base_jacobian_lens.pt", "main", 192.9),
    "qwen3-1.7b":    ("qwen3-1.7b", "Qwen3-1.7B_jacobian_lens.pt", "main", 226.5),
    "gemma-2-2b":    ("gemma-2-2b", "gemma-2-2b_jacobian_lens.pt", "main", 265.4),
    "gpt-oss-20b":   ("gpt-oss-20b", "gpt-oss-20b_jacobian_lens.pt", "main", 381.6),
    # The walkthrough's own lens: 1000 sequences, the fit the paper reports.
    "qwen3.5-4b":    ("qwen3.5-4b", "Qwen3.5-4B_jacobian_lens_n1000.pt", "qwen-n1000", 406.3),
    "gemma-3-4b":    ("gemma-3-4b", "gemma-3-4b-pt_jacobian_lens.pt", "main", 432.5),
    "qwen3-4b":      ("qwen3-4b", "Qwen3-4B_jacobian_lens.pt", "main", 458.8),
    "llama3.1-8b":   ("llama3.1-8b", "Llama-3.1-8B_jacobian_lens.pt", "main", 1040.2),
    "olmo-3-7b":     ("olmo-3-1025-7b", "Olmo-3-1025-7B_jacobian_lens.pt", "main", 1040.2),
    "gemma-2-9b":    ("gemma-2-9b", "gemma-2-9b_jacobian_lens.pt", "main", 1053.3),
    "qwen3-8b":      ("qwen3-8b", "Qwen3-8B_jacobian_lens.pt", "main", 1174.4),
    "qwen3-14b":     ("qwen3-14b", "Qwen3-14B_jacobian_lens.pt", "main", 2044.7),
    "qwen3.5-27b":   ("qwen3.5-27b", "Qwen3.5-27B_jacobian_lens.pt", "main", 3303.0),
    "qwen3.6-27b":   ("qwen3.6-27b", "Qwen3.6-27B_jacobian_lens_n1000.pt", "qwen-n1000", 3303.0),
    "gemma-3-27b":   ("gemma-3-27b", "gemma-3-27b-pt_jacobian_lens.pt", "main", 3526.0),
    "qwen3-32b":     ("qwen3-32b", "Qwen3-32B_jacobian_lens.pt", "main", 6606.0),
    "llama3.3-70b":  ("llama3.3-70b-it", "Llama-3.3-70B-Instruct_jacobian_lens.pt", "main",
                      10603.2),
}


def repo_path(key: str) -> str:
    """The path of ``key``'s lens inside the mirror repository."""
    sub, fn, _rev, _mb = CATALOG[key]
    return f"{sub}/{BASE}/{fn}"


def fetch(key: str, into: Path) -> Path:
    """Download one published lens and return the path it landed at.

    Refuses an unknown key rather than guessing a near match: the keys name specific fits, and a
    fetch that quietly returned a different model's transport would put an unrelated matrix under
    the heading the caller asked for.
    """
    if key not in CATALOG:
        raise ValueError(f"unknown lens key {key!r}; known keys: {', '.join(CATALOG)}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:                                       # pragma: no cover - env
        raise ImportError(
            "fetching a published lens needs huggingface_hub: "
            "pip install 'entroptics-jlens[lens]'") from exc
    _sub, _fn, rev, _mb = CATALOG[key]
    into.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=REPO, filename=repo_path(key), revision=rev,
                                local_dir=into))
